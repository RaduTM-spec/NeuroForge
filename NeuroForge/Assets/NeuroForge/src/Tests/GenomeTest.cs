using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using static UnityEngine.ParticleSystem;

public class GenomeTest : MonoBehaviour
{
    public Genome genome;

    private void Awake()
    {
        if (genome == null)
            genome = new Genome(2, new int[] { 2 }, ActionType.Discrete, false, true);
    }

    private void FixedUpdate()
    {
        if(Functions.RandomValue() < 0.02f)
        {
            genome.Mutate();
            //var outs = genome.Forward(new double[] { Functions.RandomValue(), Functions.RandomValue()});
            //Functions.Print(outs);
            EditorUtility.SetDirty(genome);
            AssetDatabase.SaveAssetIfDirty(genome);
        }
    }

    private void OnDrawGizmos()
    {
        if (!genome) return;

        Dictionary<NodeGene, Vector3> node_pos= new Dictionary<NodeGene, Vector3>();

        Dictionary<float, List<NodeGene>> layer_nodes = new Dictionary<float, List<NodeGene>>();
        foreach (var node in genome.nodes.Values)
        {
            if(layer_nodes.ContainsKey(node.layer))
                layer_nodes[node.layer].Add(node);
            else
                layer_nodes.Add(node.layer, new List<NodeGene> { node});
        }

        float NODE_SCALE = (4f / genome.nodes.Count);
        const float X_SCALE = 10f;
        const float Y_INC = 2f;

        // Insert inputs
        float y_pos = -Y_INC;
        foreach (var inp in layer_nodes[0])
        {
            node_pos.Add(inp, new Vector3(inp.layer * X_SCALE, y_pos, 0f));
            y_pos += Y_INC;
        }

        // Insert outputs
        y_pos = 0f;
        foreach (var outp in layer_nodes[1])
        {
            node_pos.Add(outp, new Vector3(outp.layer * X_SCALE, y_pos, 0f));
            y_pos += Y_INC;
        }

        var layers = genome.layers;
        
        for (int i = 1; i < layers.Count - 1; i++)
        {
            var hids_on_this_layer = layer_nodes[layers[i]];
            foreach (var hidden_node in hids_on_this_layer)
            {
                // get the average Y position of all previous nodes connected with 
                Dictionary<NodeGene, Vector3> previous_nodes = node_pos.Where(x => x.Key.layer <= layers[i - 1]).ToDictionary(x => x.Key, x => x.Value);

                List<ConnectionGene> incomingCons = new List<ConnectionGene>();
                hidden_node.incomingConnections.ForEach(x => incomingCons.Add(genome.connections[x]));

                List<int> inNeurons_of_incomingCons = incomingCons.Select(x => x.inNeuron).ToList();
                List<Vector3> incoming_nodes_pos = previous_nodes.Where(x =>
                        Functions.IsValueIn(x.Key.id, inNeurons_of_incomingCons)).Select(x => x.Value).ToList();

                float avg_Y = incoming_nodes_pos.Average(x => x.y);
                node_pos.Add(hidden_node, new Vector3(hidden_node.layer * X_SCALE, avg_Y, 0f));
            }
        }

        


        // Draw nodes
        foreach (var node in node_pos)
        {
            switch (node.Key.type)
            {
                case NEATNodeType.input:
                    Gizmos.color = Color.yellow;
                    break;
                case NEATNodeType.hidden:
                    Gizmos.color = Color.green;
                    break;
                case NEATNodeType.output:
                    Gizmos.color = Color.red;
                    break;
                case NEATNodeType.bias:
                    Gizmos.color = Color.blue;
                    break;
            }
            Gizmos.DrawCube(new Vector3(node.Value.x, node.Value.y, node.Value.z), Vector3.one * NODE_SCALE);

        }

        // Draw Connections
        Gizmos.color = Color.white;
        foreach (var connection in genome.connections)
        {
            Gizmos.color = connection.Value.weight < 0 ?
                                new Color(-connection.Value.weight, 0, 0) :
                                new Color(0, 0, connection.Value.weight);
            Gizmos.color = connection.Value.enabled == false ? Color.white : Gizmos.color;

            Vector3 left_right_offset = new Vector3(.5f * NODE_SCALE, 0, 0);
            Vector3 firstPoint = node_pos.Where(x => x.Key.id == connection.Value.inNeuron).Select(x => x.Value).FirstOrDefault() + left_right_offset;
            Vector3 secondPoint = node_pos.Where(x => x.Key.id == connection.Value.outNeuron).Select(x => x.Value).FirstOrDefault() - left_right_offset;

            
            Gizmos.DrawRay(firstPoint, secondPoint - firstPoint);

            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(firstPoint, NODE_SCALE * 0.1f);
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(secondPoint, NODE_SCALE * 0.1f);
        }
    }
}
