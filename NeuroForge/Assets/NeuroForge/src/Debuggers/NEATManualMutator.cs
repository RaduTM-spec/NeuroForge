using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using static UnityEngine.ParticleSystem;

public class NEATManualMutator : MonoBehaviour
{
    [Header("1. Add Connection\n" +
        "2. Mutate Node\n" +
        "3. Remove random Connection\n" +
        "4. Merge connections\n" +
        "5. Add node\n" +
        "6. Mutate connections\n")]

    public NEATNetwork mainModel;

    private void Awake()
    {
        if (mainModel == null)
        {
            mainModel = new NEATNetwork(2, new int[] { 2 }, ActionType.Discrete, false, true) ;
        }
    }

    // Update is called once per frame
    private void Start()
    {
        NEATAgent agent = new NEATAgent();
        agent.hp = new NEATHyperParameters();
        agent.hp.episodeLength = 100_000;
        agent.model = this.mainModel;
        NEATTrainer.Initialize(agent);
    }
    void Update()
    {
        double[] inputs = new double[mainModel.GetInputsNumber()];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = FunctionsF.RandomValue() < .5f ? FunctionsF.RandomValue() * -1f :
                                                        FunctionsF.RandomValue() * 1f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            mainModel.AddConnection();
       
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha2))
        {
            mainModel.MutateNode();
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha3))
        {
            mainModel.RemoveRandomConnection();
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha4))
        {
            mainModel.MergeConnections();
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha5))
        {
            mainModel.AddNode();
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha6))
        {
            mainModel.MutateConnections();
            mainModel.GetDiscreteActions(inputs);
        }
    }
    private void OnDrawGizmos()
    {
        // Draw the mainModel
        if (!mainModel) return;

        List<NodeGene> in_b = mainModel.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.input || x.type == NEATNodeType.bias).ToList();
        List<NodeGene> outp = mainModel.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.output).ToList();
        List<NodeGene> hids = mainModel.nodes.Select(x => x.Value).Where(x => x.type == NEATNodeType.hidden).ToList();

        const float SIZE_SCALE = 1f;
        const float X_SCALE = 10f;
        const float Y_INC = 2f;

        Gizmos.color = Color.grey;
        Dictionary<NodeGene, Vector3> nodesPositions = new Dictionary<NodeGene, Vector3>();

        // Compute inputs positions
        float y_pos = -Y_INC;
        foreach (var inp in in_b)
        {
            nodesPositions.Add(inp, new Vector3(0, y_pos, 0));
            y_pos += Y_INC;       
        }
        // Compute hidden positions
        y_pos = 0;
        foreach (var hid in hids)
        {
            nodesPositions.Add(hid, new Vector3(.5f * X_SCALE, y_pos, 0));
            y_pos += Y_INC;
        }
        // Compute outputs positions
        y_pos = 0;
        foreach (var inp in outp)
        {
            nodesPositions.Add(inp, new Vector3(1f * X_SCALE, y_pos, 0));
            y_pos += Y_INC;
        }

        //Draw nodes
        foreach (var node in nodesPositions)
        {
            switch(node.Key.type)
            {
                case NEATNodeType.input:
                    Gizmos.color = Color.magenta;
                    break;
                case NEATNodeType.hidden:
                    Gizmos.color = Color.yellow;
                    break;
                case NEATNodeType.output:
                    Gizmos.color = Color.red;
                    break;
                case NEATNodeType.bias:
                    Gizmos.color = Color.green;
                    break;
            }
            Gizmos.DrawCube(new Vector3(node.Value.x, node.Value.y, node.Value.z), Vector3.one * SIZE_SCALE);
        }

        //Draw connections
        Gizmos.color = Color.white;
        foreach (var connection in mainModel.connections)
        {
            Gizmos.color = connection.Value.weight < 0 ?
                                new Color(-connection.Value.weight, 0, 0) :
                                new Color(0, 0, connection.Value.weight);
            Gizmos.color = connection.Value.enabled == false ? Color.white : Gizmos.color;
            Vector3 firstPoint = nodesPositions.Where(x => x.Key.innovation == connection.Value.inNeuron).Select(x => x.Value).FirstOrDefault();
            Vector3 secondPoint = nodesPositions.Where(x => x.Key.innovation == connection.Value.outNeuron).Select(x => x.Value).FirstOrDefault();
            if (!firstPoint.Equals(secondPoint))
                Gizmos.DrawRay(firstPoint, secondPoint - firstPoint);
            else
                Gizmos.DrawWireSphere(firstPoint, SIZE_SCALE);
            
        }
    }


}
