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
        Assert(TestAddConnection);
        Assert(TestAssignRandomWeights);
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
        agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
        agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
        return true;
        
    }
    bool TestAddConnection()
    {
        
        try
        {
            NEATAgent agent = new NEATAgent();
            agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 100; i++)
            {
                agent.model.AddConnection();
            }
            NEATTrainer.Dispose();
        }
        catch
        {
            return false;
        }
        

        return true;
    }
    bool TestAssignRandomWeights()
    {
        NEATAgent agent = new NEATAgent();
        agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
        agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
        NEATTrainer.Initialize(agent);
        for (int i = 0; i < 30; i++)
        {
            agent.model.AddConnection();
        }
        /*for (int i = 0; i < 30; i++)
        {
            agent.model.AssignRandomWeights();
        }*/

        return true;

    }
    bool TestRemoveConnection()
    {  
            NEATAgent agent = new NEATAgent();
            agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
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
       
        return true;
    }
    bool TestMergeConnections()
    {
        try
        {
            NEATAgent agent = new NEATAgent();
            agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
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
            NEATTrainer.Dispose ();
            AssetDatabase.RemoveObjectFromAsset(agent.model);
        }
        catch
        {
            return false;
        }
        return true;
    }
    bool TestAddNodeToConnection()
    {
        try
        {
            NEATAgent agent = new NEATAgent();
            agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 50; i++)
            {
                agent.model.AddConnection();
            }
            for (int i = 0; i < 50; i++)
            {
                agent.model.AddNodeToConnection();
            }
            NEATTrainer.Dispose();

        }
        catch
        {
            return false;
        }
        return true;
    }
    bool TestMutateNode()
    {
        try
        {
            NEATAgent agent = new NEATAgent();
            agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);
            for (int i = 0; i < 30; i++)
            {
                agent.model.AddConnection();
            }
            for (int i = 0; i < 30; i++)
            {
                agent.model.AddNodeToConnection();
            }
            for (int i = 0; i < 100; i++)
            {
                agent.model.MutateNode();
            }
            NEATTrainer.Dispose();

        }
        catch
        {
            return false;
        }
        return true;
    }

    bool TestRandomMutations()
    {
       
        for (int k = 0; k < 5; k++)
        {
            NEATAgent agent = new NEATAgent();
            agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
            agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, false);
            NEATTrainer.Initialize(agent);


            for (int i = 0; i < 175; i++)
            {
                Debug.Log("Mutation: " + i + " | Conns: " + agent.model.connections.Count + " | Nodes: " + agent.model.nodes.Count);
                agent.model.Mutate();
            }

            NEATTrainer.Dispose();
        }
        
         
        return true;
         
         
    }
    bool TestDistance()
    {

        NEATAgent agent = new NEATAgent();
        agent.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
        agent.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, true);
        NEATTrainer.Initialize(agent);
        for (int i = 0; i < 50; i++)
        {
            agent.model.Mutate();
        }
        NEATTrainer.Dispose();


        /*NEATAgent agent2 = new NEATAgent();
        agent2.SetOnEpisodeEndType(OnEpisodeEndType.ResetNone);
        agent2.model = new NEATNetwork(2, new int[1] { 2 }, ActionType.Continuous, true);
*/


        /* NEATTrainer.Initialize(agent2);
         for (int i = 0; i < 50; i++)
         {
             agent2.model.Mutate();
         }

         NEATTrainer.InitializeHyperParameters();
         Debug.Log("Distance: " + NEATTrainer.Distance(agent.model, agent2.model));
         */

        return true;




        // AssetDatabase.RemoveObjectFromAsset(agent.model);
        // AssetDatabase.RemoveObjectFromAsset(agent2.model);

    }
    bool TestCrossover()
    {
        return true;
    }
}
