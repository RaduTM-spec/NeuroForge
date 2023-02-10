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
            mainModel = new NEATNetwork(2, new int[] { 2 }, ActionType.Discrete, true);
        }
    }

    // Update is called once per frame
    private void Start()
    {
        NEATAgent agent = new NEATAgent();
        agent.hp = new NEATHyperParameters();
        agent.hp.maxEpsiodeLength = 100_000;
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
            mainModel.ForceMutate(mainModel.AddConnection);
       
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha2))
        {
            mainModel.ForceMutate(mainModel.MutateRandomNode);
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha3))
        {
            mainModel.ForceMutate(mainModel.RemoveRandomConnection);
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha4))
        {
            mainModel.ForceMutate(mainModel.MergeConnections);
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha5))
        {
            mainModel.ForceMutate(mainModel.AddNode);
            mainModel.GetDiscreteActions(inputs);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha6))
        {
            mainModel.ForceMutate(mainModel.MutateConnections);
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
        Dictionary<int, Vector3> nodesPositions = new Dictionary<int, Vector3>();
        // Compute inputs positions
        float y_pos = 0;
        foreach (var inp in in_b)
        {
            nodesPositions.Add(inp.innovation, new Vector3(0, y_pos, 0));
            y_pos += Y_INC;
        }
        // Compute hidden positions
        y_pos = 0;
        foreach (var hid in hids)
        {
            nodesPositions.Add(hid.innovation, new Vector3(.5f * X_SCALE, y_pos, 0));
            y_pos += Y_INC;
        }
        // Compute outputs positions
        y_pos = 0;
        foreach (var inp in outp)
        {
            nodesPositions.Add(inp.innovation, new Vector3(1f * X_SCALE, y_pos, 0));
            y_pos += Y_INC;
        }

        //Draw nodes
        foreach (var node in nodesPositions)
        {
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
            Vector3 firstPoint = nodesPositions[connection.Value.inNeuron];
            Vector3 secondPoint = nodesPositions[connection.Value.outNeuron];
            if (!firstPoint.Equals(secondPoint))
                Gizmos.DrawRay(firstPoint, secondPoint - firstPoint);
            else
                Gizmos.DrawWireSphere(firstPoint, SIZE_SCALE);
            
        }
    }


}
