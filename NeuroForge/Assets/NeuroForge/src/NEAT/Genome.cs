using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class Genome : ScriptableObject, ISerializationCallbackReceiver, ICloneable
    {
        [SerializeField] public ActionType actionSpace;
        [SerializeField] public int[] outputShape;

        public Dictionary<int, NodeGene> nodes;
        public Dictionary<int, ConnectionGene> connections;

        public List<NodeGene> input_cache;
        public List<NodeGene> output_cache;

        [SerializeField] public List<float> layers;

        [SerializeField] List<NodeGene> serialized_nodes;
        [SerializeField] List<int> serialized_connections_keys;
        [SerializeField] List<ConnectionGene> serialized_connections_values;


        public Genome(int inputSize, int[] outputShape, ActionType actionSpace, bool fullyConnected, bool createAsset)
        {
            this.actionSpace = actionSpace;
            this.outputShape = outputShape;

            layers = new List<float>();
            layers.Add(0);
            layers.Add(1);

            

            nodes = new Dictionary<int, NodeGene>();
            NodeGene bias = new NodeGene(GetNextNodeId(), NEATNodeType.bias, 0);
            nodes.Add(bias.id, bias);

            input_cache = new List<NodeGene>();
            output_cache = new List<NodeGene>();

            for (int i = 0; i < inputSize; i++)
            {
                NodeGene newInput = new NodeGene(GetNextNodeId(), NEATNodeType.input, 0);
                nodes.Add(newInput.id, newInput);
                input_cache.Add(newInput);
            }
            for (int i = 0; i < outputShape.Sum(); i++)
            {
                NodeGene newOutput = new NodeGene(GetNextNodeId(), NEATNodeType.output, 1);
                nodes.Add(newOutput.id, newOutput);
                output_cache.Add(newOutput);
            }


            connections = new Dictionary<int, ConnectionGene>();
            int inov = 1;
            if (fullyConnected)
            {
                foreach (var inp in input_cache)
                {
                    foreach (var outp in output_cache)
                    {
                        ConnectionGene newConn = new ConnectionGene(inp, outp, inov++);
                        connections.Add(newConn.innovation, newConn);
                    }
                }

            }


            if (createAsset)
                CreateAsset();
        }
        public Genome() { }   
        public void SetFrom(Genome other)
        {
            this.actionSpace = other.actionSpace;
            this.outputShape = other.outputShape.ToArray();
            this.nodes = new Dictionary<int, NodeGene>();
            foreach (var nod in other.nodes)
            {
                this.nodes.Add(nod.Key, nod.Value.Clone() as NodeGene);
            }
            this.connections = new Dictionary<int, ConnectionGene>();
            foreach (var con in other.connections)
            {
                this.connections.Add(con.Key, con.Value.Clone() as ConnectionGene);
            }
            this.layers = other.layers.ToList();
            this.input_cache = this.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.input).ToList();
            this.output_cache = this.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.output).ToList();
        }
        public object Clone()
        {
            Genome clone = new Genome();

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
            clone.layers = this.layers.ToList();
            clone.input_cache = clone.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.input).ToList();
            clone.output_cache = clone.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.output).ToList();

            return clone;
        }


        // Serialization
        private void CreateAsset()
        {
            short id = 1;
            try
            {
                while (AssetDatabase.LoadAssetAtPath<NEATNetwork>("Assets/Genome#" + id + ".asset") != null)
                    id++;
            }
            catch { }

            string assetName = "Genome#" + id + ".asset";

            AssetDatabase.CreateAsset(this, "Assets/" + assetName);
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
            Debug.Log(assetName + " was created!");
        }
        public void OnBeforeSerialize()
        {
            serialized_nodes = new List<NodeGene>();
            serialized_connections_keys = new List<int>();
            serialized_connections_values = new List<ConnectionGene>();

            foreach (var node in nodes.Values)
            {
                serialized_nodes.Add(node);
            }
            foreach (KeyValuePair<int, ConnectionGene> conn in connections)
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
            input_cache = new List<NodeGene>();
            output_cache = new List<NodeGene>();
            foreach (var node in serialized_nodes)
            {
                nodes.Add(node.id, node);
                if (node.type == NEATNodeType.input)
                    input_cache.Add(node);
                else if(node.type == NEATNodeType.output)
                    output_cache.Add(node);
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
            if (actionSpace != ActionType.Discrete)
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
        public float[] Forward(double[] inputs)
        {
            // Insert inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                input_cache[i].OutValue = (float)inputs[i];
            }

            foreach (var lay in layers)
            {
                if (lay == 0)
                    continue;

                foreach (var node in nodes.Values)
                {
                    if (node.layer != lay)
                        continue;

                    // Engage this node
                    List<ConnectionGene> incommingConnections = node.incomingConnections.Select(x => connections[x]).ToList();
                    List<NodeGene> incommingNodes = incommingConnections.Select(x => nodes[x.inNeuron]).ToList();

                    float sum = 0f;
                    for (int i = 0; i < incommingNodes.Count; i++)
                    {
                        if (!incommingConnections[i].enabled)
                            continue;

                        sum += incommingNodes[i].OutValue * incommingConnections[i].weight;
                    }

                    node.InValue = sum;
                    node.Activate();

                }
            }


            // Collect outputs
            List<float> outputs = new List<float>();
            foreach (var out_node in output_cache)
            {
                outputs.Add(out_node.OutValue);
            }
            return outputs.ToArray();
        }

        // Mutations
        public void Mutate()
        {
            if (connections.Count == 0)
                AddConnection();

            float prob_mutate_cons = NEATTrainer.GetHyperParam().mutateConnections;
            float prob_mutate_node = NEATTrainer.GetHyperParam().mutateNode;
            float prob_add_conn = NEATTrainer.GetHyperParam().addConnection;
            float prob_add_node = NEATTrainer.GetHyperParam().addNode;

            if (FunctionsF.RandomValue() < prob_mutate_cons)
                MutateConnections();

            if (FunctionsF.RandomValue() < prob_mutate_node)
                MutateNode();

            if (FunctionsF.RandomValue() < prob_add_conn)
                AddConnection();

            if(FunctionsF.RandomValue() < prob_add_node)
                AddNode();

        }
        void AddConnection()
        {
            if (connections.Count >= NEATTrainer.GetHyperParam().maxConnections) return;

            List<NodeGene> listed_nodes = nodes.Values.ToList();
            NodeGene node1 = Functions.RandomIn(listed_nodes);
            NodeGene node2 = Functions.RandomIn(listed_nodes);

            // the network can be fully connected, this will end up with an infinite while-loop
            int max_tries = 25;
            while(!CanConnectTheseNodes(node1, node2))
            {
                node1 = Functions.RandomIn(listed_nodes);
                node2 = Functions.RandomIn(listed_nodes);

                if (max_tries-- == 0)
                    return; // failed to add connection
            }

            // node1.layer always smaller than node2.layer
            if (node1.layer > node2.layer)
                Functions.Swap(ref node1, ref node2);

            ConnectionGene newConn = new ConnectionGene(node1, node2, GetConnectionInnovation(node1, node2));
            connections.Add(newConn.innovation, newConn);

        }
        void AddNode()
        {
            if (nodes.Count >= NEATTrainer.GetHyperParam().maxNodes) return;
            if (connections.Count == 0) return; //is not the case, but for testing and other stuff

            ConnectionGene old_conn = Functions.RandomIn(connections.Values.ToList());
            old_conn.enabled = false;

            // Calculate new node layer
            float prev_node_layer = nodes[old_conn.inNeuron].layer;
            float next_node_layer = nodes[old_conn.outNeuron].layer;
            float this_node_layer = (prev_node_layer + next_node_layer) / 2f;

            if (!Functions.IsValueIn(this_node_layer, layers))
            {
                layers.Add(this_node_layer);
                layers.Sort((x, y) => x.CompareTo(y));
            }

            // Create node
            NodeGene new_node = new NodeGene(GetNextNodeId(), NEATNodeType.hidden, this_node_layer);
            nodes.Add(new_node.id, new_node);
            //new_node.activationType = ActivationTypeF.HyperbolicTangent;

            // Create connections
            NodeGene left_most_neuron = nodes[old_conn.inNeuron];
            NodeGene right_most_neuron = nodes[old_conn.outNeuron];

            ConnectionGene conn1 = new ConnectionGene(left_most_neuron, new_node, GetConnectionInnovation(left_most_neuron, new_node));
            connections.Add(conn1.innovation, conn1);

            ConnectionGene conn2 = new ConnectionGene(new_node, right_most_neuron, GetConnectionInnovation(new_node, right_most_neuron));
            connections.Add(conn2.innovation, conn2);

            conn1.weight = 1f;
            conn2.weight = old_conn.weight;

            
           
            
        }
        void MutateConnections()
        {
            // 1% to enable/disable the connection
            // 9% chance to completely change weight value
            // 90% chance to shift the weight value

            foreach (var connection in connections.Values)
            {
                float random = FunctionsF.RandomValue();

                if (random < 0.01f)
                    connection.enabled = connection.enabled == true ? false : true;
                else if (random < 0.1f)
                    connection.weight = FunctionsF.RandomGaussian();
                else
                    connection.weight = FunctionsF.RandomGaussian(connection.weight, 0.01f);
            }
        }
        void MutateNode()
        {
            // Get Random Hidden Node
            IEnumerable<NodeGene> hiddens = nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.hidden);
            if (hiddens.Count() == 0) return;

            NodeGene toMutate = Functions.RandomIn(hiddens);
            toMutate.activationType = (ActivationTypeF)(Enum.GetValues(typeof(ActivationTypeF)).Length * FunctionsF.RandomValue());
        }

        public bool CanConnectTheseNodes(NodeGene node1, NodeGene node2)
        {
    
            // nodes in the same layer cannot be connected (also works if the nodes are the same node)
            if (node1.layer == node2.layer) return false;

            // nodes already connected cannot be connected again

            int inNeur = node1.layer < node2.layer? node1.id : node2.id;
            int outNeur = node1.layer < node2.layer ? node2.id : node1.id;
            foreach (var con in connections.Values)
            {
                if (con.inNeuron == inNeur && con.outNeuron == outNeur)
                    return false;
            }

            return true;
        }
        private int GetConnectionInnovation(NodeGene from, NodeGene to) => InnovationHistory.Instance.GetInnovationNumber(from.id, to.id);
        private int GetNextNodeId() => nodes.Count == 0 ? 1 : nodes.Max(x => x.Key) + 1;


        // outside call
        public int GetLastNodeId() => nodes.Max(x => x.Key);
        public int GetLastInnovation() => connections.Count > 0 ? connections.Max(x => x.Key) : 0;
        public int GetInputsNumber() => input_cache.Count;
        public int GetOutputsNumber() => output_cache.Count;
        public int GetGenomeLength() => nodes.Count + connections.Count;

    }
}

