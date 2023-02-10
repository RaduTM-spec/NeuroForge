using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    /// <summary>
    /// The environment is public foreach agent, they are overlayed
    /// TODOs: change const values from network mutation to acces TRAINER hyper parameters (or mayb let mutate on their own ........don t really do the change honstly_)
    /// </summary>
    public sealed class NEATTrainer : MonoBehaviour
    {
        private static NEATTrainer Instance;

        [SerializeField] private List<NEATAgent> population;
        private List<Species> species;

        private NEATNetwork mainModel;
        private NEATHyperParameters hp;
        private TransformReseter trainingEnvironment;

        [SerializeField] private int agentsDead = 0;
        [SerializeField] private int episodeLength = 60;
        [SerializeField] private float episodeTimePassed = 0;

        [SerializeField] private int GENERATION = 0;


        private InnovationCounter innovationCounter;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
            }
        }
        private void Update()
        {
            if (Instance)
                Instance.episodeTimePassed += Time.deltaTime;
        }
        private void LateUpdate()
        {
            if (Instance != null && (Instance.episodeTimePassed >= Instance.episodeLength || Instance.agentsDead == Instance.population.Count))
            {
                // Update NEAT
                Instance.NEAT_Algorithm();


                EditorUtility.SetDirty(mainModel);
                AssetDatabase.SaveAssetIfDirty(mainModel);
                trainingEnvironment.Reset();
                // Revive all agents
                
                Instance.agentsDead = 0;
                Instance.episodeTimePassed = 0;
                Debug.Log("<color=#2873eb>Generation: " + ++GENERATION + " | Max Fitness: " + population.Max(x => x.GetFitness()) + "</color>");

                foreach (var agent in Instance.population)
                {
                    agent.Resurrect();
                }
            }
        }

        public static void Ready()
        {
            if (Instance)
                Instance.agentsDead++;
        }
        public static void Initialize(NEATAgent agent)
        {
            if (Instance != null)
                return;
            
            GameObject go = new GameObject("PPOTrainer");
            go.AddComponent<NEATTrainer>();

            Instance.population = new List<NEATAgent>() { agent };
            Instance.species = new List<Species>() { new Species(agent) };

            Instance.mainModel = agent.model;
            Instance.hp = agent.hp;
            Instance.episodeLength = agent.hp.maxEpsiodeLength;
            Instance.innovationCounter = new InnovationCounter(agent.model.GetHighestInnovation() + 1);

            try
            {
                Instance.InitPopulation(agent.gameObject, agent.hp.populationSize - 1);
                Instance.trainingEnvironment = new TransformReseter(agent.transform.parent);
            }
            catch { } // Is kept in try catch because on testing, agent is not initialized as a gameObject

        }
        private void InitPopulation(GameObject modelAgent, int size)
        {
            // init the agent gameobjects
            for (int i = 0; i < size; i++)
            {
                GameObject newAgent = Instantiate(modelAgent, modelAgent.transform.position, modelAgent.transform.rotation, modelAgent.transform.parent);
                NEATAgent newAgentScript = newAgent.GetComponent<NEATAgent>();
                Instance.population.Add(newAgentScript);
            }

            // Add population to the first species
            foreach (var agent in population)
            {
                species[1].Add(agent);
            }

            // init the agent's networks
            foreach (var agent in population)
            {
                agent.model = mainModel.Clone() as NEATNetwork;
                agent.model.Mutate();
            }     
        }


        private void NEAT_Algorithm()
        {
            List<int> genomesLength= new List<int>();
            //TODO
            foreach (var agent in population)
            {
                agent.model.Mutate();

                int total = agent.model.nodes.Count + agent.model.connections.Count;
                genomesLength.Add(total);

            }

            //Functions.Print(genomesLength);



            // Copy the best network inside mainModel and save it
            mainModel.SetFrom(population[0].model);
            EditorUtility.SetDirty(mainModel);
            AssetDatabase.SaveAssetIfDirty(mainModel);
        }
       

        // to convert to private
       
        public static bool AreCompatible(NEATNetwork parent1, NEATNetwork parent2)
        {
            float N = Max_GenesNumber(parent1, parent2);
            float E = Count_ExcessJoints(parent1, parent2);
            float D = Count_Disjoints(parent1, parent2);
            float W = Avg_WeightDifference(parent1, parent2);

            // Debug.Log("E = " + E + " | D = " + D + " | W = " + W + " | N = " + N);
            float distance = (Instance.hp.c1 * E / N) +
                             (Instance.hp.c2 * D / N) +
                             (Instance.hp.c3 * W);

            return distance < Instance.hp.delta;
        }

        private static int Max_GenesNumber(NEATNetwork genome1, NEATNetwork genome2)
        {
            int gen1_count = genome1.connections.Count + genome1.nodes.Count;
            int gen2_count = genome2.connections.Count + genome2.nodes.Count;
            return Mathf.Max(gen1_count, gen2_count);
        }
        private static float Avg_WeightDifference(NEATNetwork genome1, NEATNetwork genome2)
        {
            float dif = 0f;
            int matchesCount = 0;

            foreach (var conn1 in genome1.connections)
            {
                foreach (var conn2 in genome2.connections)
                {
                    if (conn1.Key == conn2.Key)
                    {
                        dif = Mathf.Abs(conn1.Value.weight - conn2.Value.weight);
                        matchesCount++;
                    }
                }
            }

            return matchesCount > 0 ? dif / matchesCount : 1_000_000;
        }
        private static int Count_ExcessJoints(NEATNetwork genome1, NEATNetwork genome2)
        {
            int excessJoints = 0;
            int highestMatch = 0;

            // Find highest match, excess joints are higher than this number
            foreach (var node1 in genome1.nodes)
            {
                foreach (var node2 in genome2.nodes)
                {
                    if (node1.Key == node2.Key)
                        highestMatch = Mathf.Max(highestMatch, node1.Key);
                }
            }
            foreach (var conn1 in genome1.connections)
            {
                foreach (var conn2 in genome2.connections)
                {
                    if (conn1.Key == conn2.Key)
                        highestMatch = Mathf.Max(highestMatch, conn1.Key);
                }
            }

            foreach (var node in genome1.nodes)
            {
                if (node.Key > highestMatch)
                    excessJoints++;
            }
            foreach (var node in genome2.nodes)
            {
                if (node.Key > highestMatch)
                    excessJoints++;
            }
            foreach (var conn in genome1.connections)
            {
                if (conn.Key > highestMatch)
                    excessJoints++;
            }
            foreach (var conn in genome2.connections)
            {
                if (conn.Key > highestMatch)
                    excessJoints++;
            }

            return excessJoints;
        }
        private static int Count_Disjoints(NEATNetwork genome1, NEATNetwork genome2)
        {
            int disJoints = 0;
            int highestMatch = 0;

            // Calculate highest match, joints are less than this
            foreach (var node1 in genome1.nodes)
            {
                foreach (var node2 in genome2.nodes)
                {
                    if (node1.Key == node2.Key)
                        highestMatch = Mathf.Max(highestMatch, node1.Key);
                }
            }
            foreach (var conn1 in genome1.connections)
            {
                foreach (var conn2 in genome2.connections)
                {
                    if (conn1.Key == conn2.Key)
                        highestMatch = Mathf.Max(highestMatch, conn1.Key);
                }
            }

            // now check for disjoints (need to be less than the highest match)
            foreach (var node1 in genome1.nodes)
            {
                bool isMatch = false;
                foreach (var node2 in genome2.nodes)
                {
                    if (node1.Key == node2.Key)
                    {
                        isMatch = true;
                        break;
                    }
                }

                if (!isMatch && node1.Key < highestMatch)
                    disJoints++;
            }
            foreach (var node2 in genome2.nodes)
            {
                bool isMatch = false;
                foreach (var node1 in genome1.nodes)
                {
                    if (node2.Key == node1.Key)
                    {
                        isMatch = true;
                        break;
                    }
                }

                if (!isMatch && node2.Key < highestMatch)
                    disJoints++;
            }
            foreach (var conn1 in genome1.connections)
            {
                bool isMatch = false;
                foreach (var conn2 in genome2.connections)
                {
                    if (conn1.Key == conn2.Key)
                    {
                        isMatch = true;
                        break;
                    }
                }
                if (!isMatch && conn1.Key < highestMatch)
                    disJoints++;
            }
            foreach (var conn2 in genome2.connections)
            {
                bool isMatch = false;
                foreach (var conn1 in genome1.connections)
                {
                    if (conn2.Key == conn1.Key)
                    {
                        isMatch = true;
                        break;
                    }
                }
                if (!isMatch && conn2.Key < highestMatch)
                    disJoints++;
            }

            return disJoints;
        }



        public static void Dispose() { Destroy(Instance.gameObject); Instance = null; }
        public static int GetInnovation() => Instance.innovationCounter.GetInnovation();
        public static void InitializeHyperParameters() => Instance.hp = new NEATHyperParameters();
    }
}

