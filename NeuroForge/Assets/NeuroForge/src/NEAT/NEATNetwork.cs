using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking.Types;
using UnityEngine.Windows;
using static UnityEngine.UIElements.UxmlAttributeDescription;

namespace NeuroForge
{
    [Serializable]
    public class NEATNetwork : ScriptableObject, ISerializationCallbackReceiver
    {
        [SerializeField] public ActionType actionSpace;
        [SerializeField] public int[] outputShape;

                         public Dictionary<int, NodeGene> nodes;
                         public Dictionary<int, ConnectionGene> connections; 
        
        [SerializeField] public List<NodeGene> inputNodes_cache;
        [SerializeField] public List<NodeGene> outputNodes_cache;

        [SerializeField] List<NodeGene> serialized_nodes;
        [SerializeField] List<int> seralized_connections_keys;
        [SerializeField] List<ConnectionGene> serialized_connections_values;
        public NEATNetwork(int inputSize, int[] outputShape, ActionType actionSpace, bool createAsset)
        {
            this.actionSpace = actionSpace;
            this.outputShape = outputShape;

            int innov = 1;

            nodes = new Dictionary<int, NodeGene>();
            NodeGene bias = new NodeGene(innov++, NEATNodeType.bias);
            nodes.Add(bias.innovation, bias);

            inputNodes_cache = new List<NodeGene>();
            outputNodes_cache = new List<NodeGene>();
            for (int i = 0; i < inputSize; i++)
            {
                NodeGene newInput = new NodeGene(innov++, NEATNodeType.input);
                nodes.Add(newInput.innovation, newInput);
                inputNodes_cache.Add(newInput);
            }
            for (int i = 0; i < outputShape.Sum(); i++)
            {
                NodeGene newOutput = new NodeGene(innov++, NEATNodeType.output);
                nodes.Add(newOutput.innovation, newOutput);
                outputNodes_cache.Add(newOutput);
            }

            

            // Zero conns
            connections = new Dictionary<int, ConnectionGene>();
            
           /* foreach (var inp in inputNodes_cache)
            {
                foreach(var outp in outputNodes_cache)
                {
                    ConnectionGene newConn = new ConnectionGene(inp, outp, innov++);
                    connections.Add(newConn.innovation, newConn);

                }
            }*/

            if(createAsset)
                CreateAsset();
        }
        public void OnBeforeSerialize()
        {
            serialized_nodes = new List<NodeGene>();
            seralized_connections_keys = new List<int>();
            serialized_connections_values= new List<ConnectionGene>();

            foreach (var node in nodes.Values)
            {
                serialized_nodes.Add(node);
            }
            foreach (KeyValuePair<int,ConnectionGene> keyConnection in connections)
            {
                seralized_connections_keys.Add(keyConnection.Key);
                serialized_connections_values.Add(keyConnection.Value);
            }
        }
        public void OnAfterDeserialize()
        {
            connections = new Dictionary<int, ConnectionGene>();
            for (int i = 0; i < seralized_connections_keys.Count; i++)
            {
                connections.Add(seralized_connections_keys[i], serialized_connections_values[i]);
            }

            nodes = new Dictionary<int, NodeGene>();
            foreach (var node in serialized_nodes)
            {
                nodes.Add(node.innovation, node);
            }
        }




        private void CreateAsset()
        {
            short id = 1;
            try
            {
                while (AssetDatabase.LoadAssetAtPath<NEATNetwork>("Assets/NEATNetworkNN#" + id + ".asset") != null)
                    id++;
            }
            catch { }
           
            string assetName = "NEATNetworkNN#" + id + ".asset";

            AssetDatabase.CreateAsset(this, "Assets/" + assetName);
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
            Debug.Log(assetName + " was created!");
        }
        public float[] GetContinuousActions(double[] inputs)
        {
            float[] outs = ForwardPropagation(inputs);

            for (int i = 0; i < outs.Length; i++)
            {
                outs[i] = FunctionsF.Activation.HyperbolicTangent(outs[i]);
            }

            return outs;
        }
        public int[] GetDiscreteActions(double[] inputs)
        {
            float[] outs = ForwardPropagation(inputs);
            FunctionsF.Activation.SoftMax(outs);
            int[] discreteActions = new int[outputShape.Length];

            
            
            //return branched discrete actions


            return discreteActions;
        }
        private float[] ForwardPropagation(double[] inputs)
        {
            // Insert inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                inputNodes_cache[i].OutValue = (float)inputs[i];
            }

            // Propagate
            int nodesActivated = inputNodes_cache.Count + 1; // input nodes & bias are already considered activated
            Dictionary<int,bool> nodesStatus = new Dictionary<int,bool>(); // <node id, wasActivated>
            foreach (var node in nodes)
            {
                if (node.Value.type == NEATNodeType.input || node.Value.type == NEATNodeType.bias)
                    nodesStatus.Add(node.Key, true); //input/bias nodes are already activated              
                else
                    nodesStatus.Add(node.Key, false);
            }

            int maxTries = 10_000;
            while(nodesActivated < nodes.Count)
            {
                foreach (var node in nodes)
                {
                    // if node has an output value than pass
                    if (nodesStatus[node.Key] == true)
                        continue;

                    // else calculate incoming values
                    float sum = 0;
                    int dones = 0;
                    List<ConnectionGene> sequencialCons = new List<ConnectionGene>();
                    foreach (var ic in node.Value.incomingConnections)
                    {
                        ConnectionGene incomingConn = connections[ic];
                        if (incomingConn.IsSequencial())
                        {
                            sequencialCons.Add(incomingConn);
                            dones++;
                        }
                        else if (nodesStatus[node.Key] == true || incomingConn.enabled)
                        {
                            dones++;
                            sum += incomingConn.weight * nodes[incomingConn.inNeuron].OutValue;
                        }
                        else if (incomingConn.enabled == false)
                            dones++;
                        else
                            break;
                    }
                    // If not all incoming values where calculated then pass
                    if (dones < node.Value.incomingConnections.Count)
                        continue;


                    node.Value.InValue = sum;
                    node.Value.Activate();

                    // Treat sequencial weights
                    if (sequencialCons.Count > 0)
                    {
                        
                        float sequencialInput = 0f;
                        foreach (var seq in sequencialCons)
                        {
                            if (seq.enabled == false)
                                continue;
                            sequencialInput += seq.weight * node.Value.OutValue;
                        }
                        node.Value.InValue = sequencialInput;
                        node.Value.Activate();
                    }

                    nodesStatus[node.Key] = true;
                    nodesActivated++;
                }

                if(maxTries-- == 0)
                {
                    Debug.LogError("Fatal Error: Infinite loop in forward propagation");
                    break;
                }
            }

            // Collect outputs
            float[] outputs = new float[GetOutputsNumber()];
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = outputNodes_cache[i].OutValue;
            }
            return outputs;
        }


        // Based on Difference-Based Mutation Operation for Neuroevolution of Augmented Topoligies
        public delegate void Mutation();
        public void Mutate()
        {
            switch(UnityEngine.Random.value)
            {
                case < 0.10f:
                    AddConnection();
                    break;
                case < 0.25f:
                    MutateNode();
                    break;
                case < 0.40f:
                    RemoveConnection();
                    break;
                case < 0.60f:
                    MergeConnections();
                    break;
                case < 0.70f:
                    AddNode();
                    break;
                case < 0.90f:
                    MutateConnections();
                    break;
                default:
                    //No mutation
                    break;                  
            }
        }
        public void ForceMutate(Mutation mutation)
        {
            mutation();
        }

        public void AddConnection()
        {
            // 0.1 probability
            /*
             Add connection: two nodes are selected randomly, and connected to each user. The
                weight of the connection is assigned randomly in the following way: first a random
                value from [0,1] is generated, and if it is smaller then 0.5, then the weight is set as a
                normal distributed random number with a mean equal to 0 and standard deviation set
                to 0.1 (normrand(0, 0.1)). Otherwise, the weight it set close to 1, i.e., normrand(1, 0.1).
                A new innovation number is assigned to the connection*/

            List<NodeGene> input_bias_hidden = new List<NodeGene>();
            List<NodeGene> hidden_output = new List<NodeGene>();
            nodes.Values.ToList().ForEach(x =>
            {
                switch(x.type)
                {
                    case NEATNodeType.bias:
                        input_bias_hidden.Add(x);
                        break;
                    case NEATNodeType.input:
                        input_bias_hidden.Add(x);
                        break;
                    case NEATNodeType.hidden:
                        input_bias_hidden.Add(x);
                        hidden_output.Add(x);
                        break;
                    case NEATNodeType.output:
                        hidden_output.Add(x);
                        break;
                }
            });
            NodeGene neur1 = Functions.RandomIn(input_bias_hidden);
            NodeGene neur2 = Functions.RandomIn(hidden_output);

            // weights (nodes more precisely) can be sequencial
            // double connection is allowed

            ConnectionGene newConnection = new ConnectionGene(neur1, neur2, NEATTrainer.GetInnovation());

            connections.Add(newConnection.innovation, newConnection);

        } // good
        public void MutateConnections()
        {
            // 0.2 probability
            //Assigning random weights: every connection is mutated with a probability of
            //1 / NumberO f Connections the connection receives either weight, chosen from
            //normrand(0, 0.1) or normrand(1, 0.1). Otherwise, the current value of the weight
            //is used as a mean value to generate new as follows: w = normrand(w, 0.01), where w
            //is the current weight value. With probability of 1 / NumberO f Connections each weight
            //is either activated or deactivated.
            if (connections.Count == 0) return;

            foreach (var connection in connections.Values)
            {
                // Mutate weight
                if (FunctionsF.RandomValue() > 1f / connections.Count)
                {
                    // complete new value
                    connection.weight = FunctionsF.RandomValue() < .5f ?
                                        FunctionsF.RandomGaussian(0, 0.1f) :
                                        FunctionsF.RandomGaussian(1, 0.1f);
                }
                else
                {
                    // slight lerp
                    connection.weight = FunctionsF.RandomGaussian(connection.weight, 0.1f);
                }



                // Enable or disable
                if (FunctionsF.RandomValue() > 1f / connections.Count)
                {
                    connection.enabled = connection.enabled == true ? false : true;
                }


            }

        } // good
        public void RemoveConnection()
        {
            // 0.15 probability
            //Connection removal: if the structure of the network is not minimal, then a randomly chosen connection is deleted. The connections between inputs and outputs are
            //never chosen.

            IEnumerable<int> inNodes = inputNodes_cache.Select(x => x.innovation);
            IEnumerable<int> outNodes = outputNodes_cache.Select(x => x.innovation);

            // Connections between inputs and outputs are never removed
            List<ConnectionGene> removeableConns = new List<ConnectionGene>();
            connections.Values.ToList().ForEach((x) =>
            {
                if (!Functions.IsValueIn(x.inNeuron, inNodes) || !Functions.IsValueIn(x.outNeuron, outNodes))
                {
                    removeableConns.Add(x);
                }

            });

            if (removeableConns.Count == 0) return;
            ConnectionGene toRemove = Functions.RandomIn(removeableConns);

            connections.Remove(toRemove.innovation);
            foreach (var node in nodes.Values)
            {
                try
                { node.incomingConnections.Remove(toRemove.innovation); }
                catch { }      //               
            }
                 
            
         

        } // good
        public void MergeConnections()
        {
            // 0.2 probabillity
            /*Connections merging: if the network structure contains at least two connections
            following the same path, i.e., having the same source and destination, then these
            nodes are merged together, and the weight value is assigned as the sum of weights.
            The new connection receives the innovation number of one of the merged.
            */
            if (connections.Count == 0) return;

            foreach (var conn1 in connections.Values)
            {
                foreach (var conn2 in connections.Values)
                {
                    if (conn1.Equals(conn2))
                        continue;

                    if(conn1.inNeuron == conn2.inNeuron && conn1.outNeuron == conn2.outNeuron)
                    {
                        int newInnov = FunctionsF.RandomValue() < .5f? conn1.innovation : conn2.innovation;
                        ConnectionGene mergedConnection = new ConnectionGene(nodes[conn1.inNeuron], nodes[conn1.outNeuron], newInnov);
                        mergedConnection.weight = conn1.weight + conn2.weight;

                        connections.Remove(conn1.innovation);
                        connections.Remove(conn2.innovation);
                        foreach (var neuron in nodes.Values)
                        {
                            try
                            {
                                neuron.incomingConnections.Remove(conn1.innovation);
                            }
                            catch { }
                            try
                            {
                                neuron.incomingConnections.Remove(conn2.innovation);
                            }
                            catch { }
                        }

                        connections.Add(newInnov, mergedConnection);
                        return; // Only one merge for now                  
                    }
                }
            }


        } // good
        public void AddNode()
        {
            // 0.1 probability
            /*
             Adding a node to connection: one of the connections is randomly selected and divided
                into two, and a node is placed in between. The new node receives a new innovation
                number and operation, one of the weights is set to 1, while the other keeps the previous
                value. The connections receive new innovation numbers.
            */
            if (connections.Count == 0) return;

            ConnectionGene oldConnection = connections[Functions.RandomIn(connections.Keys)];
            connections.Remove(oldConnection.innovation);

            // Create splitter node
            NodeGene newNeuron = new NodeGene(NEATTrainer.GetInnovation(), NEATNodeType.hidden);
            nodes.Add(newNeuron.innovation, newNeuron);

            // Create first connection
            ConnectionGene conn1 = new ConnectionGene(nodes[oldConnection.inNeuron], newNeuron, NEATTrainer.GetInnovation());
            connections.Add(conn1.innovation, conn1);

            // Create second connection
            ConnectionGene conn2 = new ConnectionGene(newNeuron, nodes[oldConnection.outNeuron], NEATTrainer.GetInnovation());
            connections.Add(conn2.innovation, conn2);
            nodes[oldConnection.outNeuron].incomingConnections.Remove(oldConnection.innovation);

            // Assign the weights
            if (FunctionsF.RandomValue() < .5f)
            {
                conn1.weight = 1f;
                conn2.weight = oldConnection.weight;
            }
            else
            {
                conn1.weight = oldConnection.weight;
                conn2.weight = 1f;
            }




        } // good
        public void MutateNode()
        {
            // 0.15 probability
            /*Mutating random node: the type of operation performed in a randomly chosen node
                is changed to another one, and the new innovation number is assigned to the node.
                Only hidden nodes are mutating.*/

            // Get Random Hidden Node
            IEnumerable<NodeGene> hiddens = nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.hidden);

            if (hiddens.Count() == 0) return;

            NodeGene oldNode = Functions.RandomIn(hiddens);

            // TEMPORARY modification
            oldNode.activationType = (ActivationTypeF) (Enum.GetValues(typeof(ActivationTypeF)).Length * FunctionsF.RandomValue());



            return;



            NodeGene newNode = new NodeGene(NEATTrainer.GetInnovation(), NEATNodeType.hidden);
            newNode.incomingConnections = oldNode.incomingConnections.ToList();

            // Update the new innovations to all connections
            foreach (var conn in connections.Values)
            {
                // Consider sequencial nodes

                if (conn.inNeuron == oldNode.innovation)
                    conn.inNeuron = newNode.innovation;

                if (conn.outNeuron == oldNode.innovation)
                    conn.outNeuron = newNode.innovation;
            }

            nodes.Remove(oldNode.innovation);

        } // not good


        // Not from the upper paper
        public void EnableDisableConnection()
        {
            ConnectionGene randomConnection = connections[Functions.RandomIn(connections.Keys)];

            randomConnection.enabled = randomConnection.enabled == true ? false : true;
        }
        



        // Other
        public int GetHighestInnovation()
        {
            int max_nodes_inov = nodes.Keys.Max();
            int max_conec_inov = connections.Count > 0 ? connections.Keys.Max() : -1;
            return Math.Max(max_conec_inov, max_nodes_inov);
        }
        public int GetInputsNumber() => inputNodes_cache.Count;
        public int GetOutputsNumber() => outputNodes_cache.Count;

    }
}