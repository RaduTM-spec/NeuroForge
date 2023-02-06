using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using System;
using System.Linq;
using UnityEditor;
using System.Linq.Expressions;

public class NEATUnitTests : MonoBehaviour
{
    public delegate bool TestFunc();
    public List<TestFunc> tests = new List<TestFunc>();

    private void Start()
    {
        AssertAll();
        RunAll();
    }
    
    public void RunAll()
    {
        StringBuilder messages = new StringBuilder();
        messages.AppendLine();
        int count = 1;
        foreach (TestFunc test in tests)
        {
            bool res = test();
            if (res)
            {
                messages.AppendLine("<color=green>" + count + ". " + test.Method.Name + " passed!</color>");
            }
            else
            {
                messages.AppendLine("<color=red>" + count + ". " + test.Method.Name + " failed!</color>");
            }
            count++;
        }
        Debug.Log(messages.ToString());
        tests.Clear();
    }

    public void Assert(TestFunc test) => tests.Add(test);


    //------------Tests----------------//
    public void AssertAll()
    {
        Assert(TestCreateNEATNET);
        Assert(TestSequencials);
        Assert(TestAddConnection);
        Assert(TestMutateConnections);
        Assert(TestRemoveConnection);
        Assert(TestMergeConnections);
        Assert(TestAddNodeToConnection);
        Assert(TestMutateNode);
        
        Assert(TestRandomMutations);
        Assert(TestDistance);
        Assert(TestCrossover);
    }

    bool TestCreateNEATNET()
    {
        NEATAgent agent = new NEATAgent();
        agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
        return true;
        
    }
    bool TestSequencials()
    {
        NEATAgent agent = new NEATAgent();
        agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
        NEATTrainer.Initialize(agent);
        for (int i = 0; i < 1; i++)
        {
            agent.model.AddConnection();
            agent.model.AddNode();
        }
        for (int i = 0; i < 250; i++)
        {
            agent.model.AddConnection();
        }

        EditorUtility.SetDirty(agent.model);
        AssetDatabase.SaveAssetIfDirty(agent.model);
        NEATTrainer.Dispose();
        agent.model.GetContinuousActions(new double[] { 1, 1 });

        return true;
    }
    bool TestAddConnection()
    {
            NEATAgent agent = new NEATAgent();
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 100; i++)
            {
                agent.model.AddConnection();
            }
        EditorUtility.SetDirty(agent.model);
        AssetDatabase.SaveAssetIfDirty(agent.model);
            NEATTrainer.Dispose();
            agent.model.GetContinuousActions(new double[] { 1, 1 });

        return true;
    }
    bool TestMutateConnections()
    {
        NEATAgent agent = new NEATAgent();
        agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
        NEATTrainer.Initialize(agent);
        for (int i = 0; i < 30; i++)
        {
            agent.model.AddConnection();
        }
        for (int i = 0; i < 30; i++)
        {
            agent.model.MutateConnections();
        }
        NEATTrainer.Dispose();
        agent.model.GetContinuousActions(new double[] { 1, 1 });

        return true;

    }
    bool TestRemoveConnection()
    {  
            NEATAgent agent = new NEATAgent();
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 50; i++)
            {
                agent.model.AddConnection();
            }
            for (int i = 0; i < 50; i++)
            {
                agent.model.RemoveConnection();
            }
            NEATTrainer.Dispose();
        agent.model.GetContinuousActions(new double[] { 1, 1 });
        return true;
    }
    bool TestMergeConnections()
    {
        
            NEATAgent agent = new NEATAgent();
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 50; i++)
            {
                agent.model.AddConnection();
            }
            for (int i = 0; i < 50; i++)
            {
                agent.model.MergeConnections();
            }
            NEATTrainer.Dispose();
            AssetDatabase.RemoveObjectFromAsset(agent.model);
        agent.model.GetContinuousActions(new double[] { 1, 1 });
        return true;
    }
    bool TestAddNodeToConnection()
    {
       
            NEATAgent agent = new NEATAgent();
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 50; i++)
            {
                agent.model.AddConnection();
            }
            for (int i = 0; i < 50; i++)
            {
                agent.model.AddNode();
            }
            NEATTrainer.Dispose();
        agent.model.GetContinuousActions(new double[] { 1, 1 });

        return true;
    }
    bool TestMutateNode()
    {   
            NEATAgent agent = new NEATAgent();  
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 30; i++)
            {
                agent.model.AddConnection();
            }
            for (int i = 0; i < 30; i++)
            {
                agent.model.AddNode();
            }
            for (int i = 0; i < 100; i++)
            {
                agent.model.MutateNode();
            }
            NEATTrainer.Dispose();
        agent.model.GetContinuousActions(new double[] { 1, 1 });

        return true;
    }

    bool TestRandomMutations()
    {
      
        for (int k = 0; k < 25; k++)
        {
            NEATAgent agent = new NEATAgent();
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);

            List<NEATNetwork.Mutation> mutations = new List<NEATNetwork.Mutation>();
            mutations.Add(agent.model.AddConnection);
            mutations.Add(agent.model.MutateNode); // -> here is problem
            mutations.Add(agent.model.RemoveConnection);
            mutations.Add(agent.model.MergeConnections);
            mutations.Add(agent.model.AddNode);
            mutations.Add(agent.model.MergeConnections);
            
            for (int i = 0; i < 200; i++)
            {
                agent.model.ForceMutate(Functions.RandomIn(mutations));
            }

            NEATTrainer.Dispose();
        }

        return true;      
    }
    bool TestDistance()
    {
        for (int k = 5; k < 25; k++)
        {
            NEATAgent agent = new NEATAgent();
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            //StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < k; i++)
            {
                //stringBuilder.AppendLine("Mutation: " + i + " | Conns: " + agent.model.connections.Count + " | Nodes: " + agent.model.nodes.Count);

                agent.model.Mutate();
            }
            NEATTrainer.Dispose();

            NEATAgent agent2 = new NEATAgent();
            agent2.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent2);
            for (int i = 0; i < k; i++)
            {
                agent2.model.Mutate();
            }

            NEATTrainer.InitializeHyperParameters();
            //Debug.Log("Distance: " + NEATTrainer.AreCompatible(agent.model, agent2.model));

        }


        return true;

        // EditorUtility.SetDirty(models)
        // SaveAssetIfDirty(models)


        // AssetDatabase.RemoveObjectFromAsset(agent.model);
        // AssetDatabase.RemoveObjectFromAsset(agent2.model);

    }
    bool TestCrossover()
    {
        return true;
        // disjoint 
    }
}
