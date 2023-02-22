using Palmmedia.ReportGenerator.Core.Parser.Analysis;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using Unity.VisualScripting.Antlr3.Runtime;
using Unity.VisualScripting.Antlr3.Runtime.Tree;
using UnityEditor;
using UnityEditor.Experimental.GraphView;
using UnityEditor.MemoryProfiler;
using UnityEditor.Profiling;
using UnityEngine;
using UnityEngine.Networking.Types;
using UnityEngine.Windows;
using static UnityEngine.UIElements.UxmlAttributeDescription;

namespace NeuroForge
{
    [Serializable]
    public class NEATNetwork : ScriptableObject, ISerializationCallbackReceiver, ICloneable
    {
        [SerializeField] public ActionType actionSpace;
        [SerializeField] public int[] outputShape;

                         public Dictionary<int, NodeGene> nodes;
                         public Dictionary<int, ConnectionGene> connections; 
        
        [SerializeField] public List<int> inputNodes_cache;
        [SerializeField] public List<int> outputNodes_cache;

        [SerializeField] List<NodeGene> serialized_nodes;
        [SerializeField] List<int> serialized_connections_keys;
        [SerializeField] List<ConnectionGene> serialized_connections_values;
        

        // Initialize
        public NEATNetwork(int inputSize, int[] outputShape, ActionType actionSpace, bool fullyConnected, bool createAsset)
        {
            this.actionSpace = actionSpace;
            this.outputShape = outputShape;

            int innov = 1;

            nodes = new Dictionary<int, NodeGene>();
            NodeGene bias = new NodeGene(innov++, NEATNodeType.bias, 0);
            nodes.Add(bias.innovation, bias);

            inputNodes_cache = new List<int>();
            outputNodes_cache = new List<int>();
            for (int i = 0; i < inputSize; i++)
            {
                NodeGene newInput = new NodeGene(innov++, NEATNodeType.input,0);
                nodes.Add(newInput.innovation, newInput);
                inputNodes_cache.Add(newInput.innovation);
            }
            for (int i = 0; i < outputShape.Sum(); i++)
            {
                NodeGene newOutput = new NodeGene(innov++, NEATNodeType.output, 1);
                nodes.Add(newOutput.innovation, newOutput);
                outputNodes_cache.Add(newOutput.innovation);
            }

           
            connections = new Dictionary<int, ConnectionGene>();
            
            if(fullyConnected)
            {
                foreach (var inp in inputNodes_cache)
                {
                    foreach (var outp in outputNodes_cache)
                    {
                        ConnectionGene newConn = new ConnectionGene(nodes[inp], nodes[outp], innov++);
                        connections.Add(newConn.innovation, newConn);
                    }
                }
                
            }
        

            if(createAsset)
                CreateAsset();
        }
        public void SetFrom(NEATNetwork copy)
        {
            this.actionSpace= copy.actionSpace;
            this.outputShape = copy.outputShape.ToArray();
            this.nodes = new Dictionary<int, NodeGene>();
            foreach (var nod in copy.nodes)
            {
                this.nodes.Add(nod.Key, nod.Value.Clone() as NodeGene);
            }
            this.connections = new Dictionary<int, ConnectionGene>();
            foreach (var con in copy.connections)
            {
                this.connections.Add(con.Key, con.Value.Clone() as ConnectionGene);
            }
            this.inputNodes_cache = copy.inputNodes_cache.Select(x => x).ToList();
            this.outputNodes_cache = copy.outputNodes_cache.Select(x => x).ToList();
        }
        private NEATNetwork() { }
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
        public object Clone()
        {
            NEATNetwork clone = new NEATNetwork();

            clone.actionSpace = this.actionSpace;
            clone.outputShape = this.outputShape.ToArray();
            clone.nodes = new Dictionary<int, NodeGene>();
            foreach (var nod in this.nodes)
            {
                clone.nodes.Add(nod.Key, nod.Value.Clone() as NodeGene);
            }
            clone.connections = new Dictionary<int, ConnectionGene>();
            foreach (var conn in this.connections)
            {
                clone.connections.Add(conn.Key, conn.Value.Clone() as ConnectionGene);
            }

            clone.inputNodes_cache = this.inputNodes_cache.Select(x => x).ToList();
            clone.outputNodes_cache = this.outputNodes_cache.Select(x => x).ToList();

            return clone;
        }

        // Serialization
        public void OnBeforeSerialize()
        {
            serialized_nodes = new List<NodeGene>();
            serialized_connections_keys = new List<int>();
            serialized_connections_values= new List<ConnectionGene>();

            foreach (var node in nodes.Values)
            {
                serialized_nodes.Add(node);
            }
            foreach (KeyValuePair<int,ConnectionGene> conn in connections)
            {
                serialized_connections_keys.Add(conn.Key);
                serialized_connections_values.Add(conn.Value);
            }

            // Do not ever save asset dirty from here otherwise you get crash
        }
        public void OnAfterDeserialize()
        {
            connections = new Dictionary<int, ConnectionGene>();

            nodes = new Dictionary<int, NodeGene>();
            foreach (var node in serialized_nodes)
            {
                nodes.Add(node.innovation, node);
            }

            for (int i = 0; i < serialized_connections_keys.Count; i++)
            {
                connections.Add(serialized_connections_keys[i], serialized_connections_values[i]);
            }

            
        }


        // Propagation
        public float[] GetContinuousActions(double[] inputs)
        {
            if (actionSpace != ActionType.Continuous)
            {
                Debug.LogError("Cannot get continuous actions from a discrete model");
                throw new Exception("Cannot get continuous actions from a discrete model");
            }

            float[] outs = Forward(inputs);

            for (int i = 0; i < outs.Length; i++)
            {
                outs[i] = FunctionsF.Activation.HyperbolicTangent(outs[i]);
            }

            return outs;
        }
        public int[] GetDiscreteActions(double[] inputs)
        {
            if(actionSpace != ActionType.Discrete)
            {
                Debug.LogError("Cannot get discrete actions from a continuous model");
                throw new Exception("Cannot get discrete actions from a continuous model");
            }
            int[] discreteActions = new int[outputShape.Length];
            
            float[] outs = Forward(inputs);
            FunctionsF.Activation.SoftMax(outs);
            List<float> activatedOutputs = outs.ToList();

            int index = 0;
            for (int i = 0; i < outputShape.Length; i++)
            {
                float[] branchValues = activatedOutputs.GetRange(index, outputShape[i]).ToArray();
                discreteActions[i] = FunctionsF.Activation.ArgMax(branchValues);
            }
            
            return discreteActions;
        }
        private float[] Forward(double[] inputs)
        {
            // Insert inputs
            for (int i = 0; i < inputNodes_cache.Count; i++)
            {
                nodes[inputNodes_cache[i]].OutValue = (float)inputs[i];
            }

            // Deactivated all hidden and output
            foreach (var node in nodes.Values)
            {
                if (node.type == NEATNodeType.hidden || node.type == NEATNodeType.output)
                    node.Deactivate();
            }

            // Keep maxLoops as it is. 
            // If there will be the following situation:
            // node1 -> node2 && node2 -> node1
            // There will be a deadlock. In this situation is recommended to stop in this way.

            int maxLoops = nodes.Count;
            while(maxLoops-- > 0)
            {
                // Something wrong in here
                foreach (var node in nodes)
                {
                    if (node.Value.IsActivated())
                        goto NEXT_NODE;

                    if(node.Value.incomingConnections.Count == 0)
                    {
                        node.Value.Activate();
                        goto NEXT_NODE;
                    }

                    List<ConnectionGene> incoming_connections = node.Value.incomingConnections.Select(x => connections[x]).ToList();
                    Dictionary<ConnectionGene, NodeGene> incoming_nw_pairs = new Dictionary<ConnectionGene, NodeGene>();
                    incoming_connections.ForEach(x =>
                    {
                        if(!x.IsSequencial())
                           incoming_nw_pairs.Add(x, nodes[x.inNeuron]);
                    });

                    
                    // Check if all incoming nodes are done
                    foreach (var prev_node in incoming_nw_pairs.Values)
                        if (!prev_node.IsActivated())
                            goto NEXT_NODE;
                    
                    // -----------------NODE IS PREPARED TO BE ACTIVATED-----------------------//
                    // propagate value from input nodes
                    float sum = 0f;
                    bool seq_found = false;
                    foreach (var nw_pair in incoming_nw_pairs)
                    {
                        if (nw_pair.Key.IsSequencial())
                        {
                            seq_found = true;
                            continue;                          
                        }
                        if (!nw_pair.Key.enabled)
                            continue;

                        sum += nw_pair.Value.OutValue * nw_pair.Key.weight;
                    }
                    node.Value.InValue = sum;
                    node.Value.Activate();

                    // treat sequencial weights
                    if(seq_found)
                    {
                        sum = 0f;
                        foreach (var inc_con in incoming_connections)
                        {
                            if (!inc_con.IsSequencial())
                                continue;

                            if (!inc_con.enabled)
                                continue;

                            seq_found = true;
                            sum += node.Value.OutValue * inc_con.weight;
                        }
                        node.Value.InValue = sum;
                        node.Value.Activate();
                    }

                NEXT_NODE:
                    continue;
                }

                // If all nodes where activated, stop the while loop
                if (nodes.Select(x => x.Value).Where(x => !x.IsActivated()).Any() == false)
                      break;
            }
            if (maxLoops == 0)
                RemoveRandomConnection(); // if a deadlock happens, try randomly to remove the deadlock
            
            // Collect outputs
            float[] outputs = new float[GetOutputsNumber()];
            for (int i = 0; i < outputNodes_cache.Count; i++)
            {
                outputs[i] = nodes[outputNodes_cache[i]].OutValue;
            }
            return outputs;
        }


        // Mutations
        public delegate void Mutation();
        Dictionary<Mutation, float> mutations;
        public void Mutate()
        {
            if(mutations == null)
            {
                mutations = new Dictionary<Mutation, float>();
                mutations.Add(AddConnection, NEATTrainer.GetHP().addConnection);
                mutations.Add(MutateNode, NEATTrainer.GetHP().mutateNode);
                mutations.Add(RemoveRandomConnection, NEATTrainer.GetHP().removeConnection);
                mutations.Add(MergeConnections, NEATTrainer.GetHP().mergeConnections);
                mutations.Add(AddNode, NEATTrainer.GetHP().addNode);
                mutations.Add(MutateConnections, NEATTrainer.GetHP().mutateConnections);
                mutations.Add(NoMutation, NEATTrainer.GetHP().noMutation);

                // mutations mustn't necesarrily be sorted in ascending order based on the probabilities
            }

            // Select Random Mutation and execute it
            float random = FunctionsF.RandomValue();
            foreach (var mut in mutations)
            {
                random -= mut.Value;
                if (random <= 0)
                {
                    mut.Key();
                    return;
                }                
            }
           
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

            if (connections.Count >= NEATTrainer.GetHP().maxConnections) return;

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
            // deadlock is allowed (and must be confronted)

            ConnectionGene newConnection = new ConnectionGene(neur1, neur2, NEATTrainer.GetInnovation());

            connections.Add(newConnection.innovation, newConnection);

        } 
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

            // On short
            // 1/C chance to complete new value
            foreach (var connection in connections.Values)
            {
                // Mutate weight
                if (FunctionsF.RandomValue() < 1f / connections.Count)
                {
                    connection.weight = FunctionsF.RandomGaussian();
                    // complete new value

                    // Paper alternative (is changed because i want weights to be also strong negative
                   /* connection.weight = FunctionsF.RandomValue() < .5f ?
                                        FunctionsF.RandomGaussian(0, 0.1f) :
                                        FunctionsF.RandomGaussian(1, 0.1f);*/
                }
                else
                {
                    // slight lerp
                    connection.weight = FunctionsF.RandomGaussian(connection.weight, 0.1f);
                }



                // Enable or disable
                if (FunctionsF.RandomValue() < 1f / connections.Count)
                {
                    connection.enabled = connection.enabled == true ? false : true;
                }


            }

        } 
        public void RemoveRandomConnection()
        {
            // 0.15 probability
            //Connection removal: if the structure of the network is not minimal, then a randomly chosen connection is deleted. The connections between inputs and outputs are
            //never chosen.

            // Connections between inputs and outputs are never removed
            List<ConnectionGene> removeableConns = new List<ConnectionGene>();
            connections.Values.ToList().ForEach((x) =>
            {
                if (!Functions.IsValueIn(x.inNeuron, inputNodes_cache) || !Functions.IsValueIn(x.outNeuron, outputNodes_cache))
                {
                    removeableConns.Add(x);
                }

            });

            if (removeableConns.Count == 0) return;
            ConnectionGene toRemove = Functions.RandomIn(removeableConns);

            connections.Remove(toRemove.innovation);
            foreach (var node in nodes.Values)
            {
                if(Functions.IsValueIn(toRemove.innovation, node.incomingConnections))
                    node.incomingConnections.Remove(toRemove.innovation);            
            }
                 
        } 
        public void MergeConnections()
        {
            // 0.2 probabillity
            /*Connections merging: if the network structure contains at least two connections
            following the same path, i.e., having the same source and destination, then these
            nodes are merged together, and the weight value is assigned as the sum of weights.
            The new connection receives the innovation number of one of the merged.
            */
            if (connections.Count == 0) return;

            bool found = true;
            while(found)
            {
                found = false;

                List<ConnectionGene> cons = connections.Values.ToList();
                foreach (var con1 in cons)
                {
                    foreach (var con2 in cons)
                    {
                        if (con1.innovation == con2.innovation)
                            continue;

                        if (con1.inNeuron != con2.inNeuron || con1.outNeuron != con2.outNeuron)
                            continue;

                        int newInov = FunctionsF.RandomValue() < .5f? con1.innovation: con2.innovation;
                        ConnectionGene mergedCon = new ConnectionGene(nodes[con1.inNeuron], nodes[con1.outNeuron], newInov);
                        mergedCon.weight = con1.weight + con2.weight;

                        connections.Remove(con1.innovation);
                        connections.Remove(con2.innovation);

                        NodeGene outNeur = nodes[con1.outNeuron];
                        if (outNeur.incomingConnections.Remove(con1.innovation) == false)
                            Debug.LogError("Problem here");
                        if (outNeur.incomingConnections.Remove(con2.innovation) == false)
                            Debug.LogError("Problem here2");

                        connections.Add(newInov, mergedCon);
                        found = true;
                        goto FIND_AGAIN;
                    }
                }
                FIND_AGAIN:            
                continue;
            }
            


        } 
        public void AddNode()
        {
            // 0.1 probability
            /*
             Adding a node to connection: one of the connections is randomly selected and divided
                into two, and a node is placed in between. The new node receives a new innovation
                number and operation, one of the weights is set to 1, while the other keeps the previous
                value. The connections receive new innovation numbers.
            */
            // Max hidden allowed = inputs + outputs
            if (connections.Count == 0) return;
            if (nodes.Count >= NEATTrainer.GetHP().maxNodes) return;

            ConnectionGene oldConnection = connections[Functions.RandomIn(connections.Keys)];
            connections.Remove(oldConnection.innovation);



            // Calculate layer to place the node
            float prev_lay = nodes[oldConnection.inNeuron].layer;
            float next_lay = nodes[oldConnection.outNeuron].layer;
            float layer_to_place = (prev_lay + next_lay) / 2;
       
            // Create splitter node        
            NodeGene newNeuron = new NodeGene(NEATTrainer.GetInnovation(), NEATNodeType.hidden, layer_to_place);
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




        } 
        public void MutateNode()
        {
            // 0.15 probability
            /*Mutating random node: the type of operation performed in a randomly chosen node
                is changed to another one, and the new innovation number is assigned to the node.
                Only hidden nodes are mutating.*/
            // In my implementation, the innovation remains the same, only the activation is changed

            // Get Random Hidden Node
            IEnumerable<NodeGene> hiddens = nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.hidden);
            if (hiddens.Count() == 0) return;

            NodeGene toMutate = Functions.RandomIn(hiddens);
            toMutate.activationType = (ActivationTypeF)(Enum.GetValues(typeof(ActivationTypeF)).Length * FunctionsF.RandomValue());
        }
        public void NoMutation() { }



        // Other
        public override string ToString()
        {
            StringBuilder nodesSB = new StringBuilder("nodes->");
            StringBuilder connectionsSB = new StringBuilder("\nconnections->");
            foreach (var node in nodes)
            {
                nodesSB.Append(node.Value.ToString());
            }
            foreach (var item in connections)
            {
                connectionsSB.Append(item.Value.ToString());
            }
            nodesSB.Append(connectionsSB.ToString());
            return nodesSB.ToString();

        }
        public int GetHighestInnovation()
        {

            int max_nodes_inov = nodes.Keys.Max();
            int max_conec_inov = connections.Count > 0 ? connections.Keys.Max() : -1;

           
            return Math.Max(max_conec_inov, max_nodes_inov);
        }
        public int GetInputsNumber() => inputNodes_cache.Count;
        public int GetOutputsNumber() => outputNodes_cache.Count;
        public int GetGenomeLength() => nodes.Count + connections.Count;

    }
}