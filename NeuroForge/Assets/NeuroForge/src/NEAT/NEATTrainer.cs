using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    /// <summary>
    /// The environment is public foreach agent, they are overlayed
    /// TODOs: check why ends up having 51 members in 2 species instead of a total of 50 (one is added i don t know from where
    /// Decide between protecting a species (and having a maximum number of species) by no killing if too small, or use GoExtinct
    /// </summary>
    public sealed class NEATTrainer : MonoBehaviour
    {
        private static NEATTrainer Instance;

        [SerializeField] private HashSet<NEATAgent> population;
        private HashSet<Species> species;

        private NEATNetwork mainModel;
        private NEATHyperParameters hp;
        private TransformReseter trainingEnvironment;

        [SerializeField] private int agentsDead = 0;
        [SerializeField] private float episodeTimePassed = 0;

        [SerializeField] private int GENERATION = 0;
        [SerializeField] private bool SESSION_END = false;

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
            if (!SESSION_END && Instance != null && (Instance.episodeTimePassed >= Instance.hp.episodeLength || Instance.agentsDead == Instance.population.Count))
            {
                // Update NEAT
                Instance.Evolution();

                // Reset Environment
                Instance.trainingEnvironment.Reset();

                // Reset Episode Stats
                Instance.agentsDead = 0;
                Instance.episodeTimePassed = 0;

                // Print Episode Statistic
                Debug.Log(GetEpisodeStatistic());

                // Resurrect agents
                foreach (var agent in Instance.population)
                {
                    agent.Resurrect();
                    agent.ResetFitness();
                }

                // Check for stop
                if(GENERATION == hp.generations)
                {
                    SESSION_END = true;
                    species = null;
                    foreach (var ag in population)
                    {
                        ag.behaviour = BehaviourType.Inactive;
                    }
                    Debug.Log("<color=green> Training session ended! </color>");
                    EditorApplication.isPlaying = false;
                   
                }
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
                switch (node.Key.type)
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

        // Trainer
        public static void Ready()
        {
            if (Instance)
                Instance.agentsDead++;
        }
        public static void Initialize(NEATAgent agent)
        {
            if (Instance != null)
                return;
            
            GameObject go = new GameObject("NEATTrainer");
            go.AddComponent<NEATTrainer>();

            Instance.population = new HashSet<NEATAgent>() { agent };
            Instance.species = new HashSet<Species>();

            Instance.mainModel = agent.model;
            Instance.hp = agent.hp;
            Instance.hp.generations = agent.hp.generations;
            Instance.hp.episodeLength = agent.hp.episodeLength;
            Instance.innovationCounter = new InnovationCounter(agent.model.GetHighestInnovation() + 1);

            
            Instance.InitPopulation(agent.gameObject, agent.hp.populationSize - 1);
            Instance.trainingEnvironment = new TransformReseter(agent.transform.parent); // is ok placed here, to get reference of all other agents
            Instance.Evolution();
          
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

            // init the agent's networks
            foreach (var agent in population)
            {
                agent.model = mainModel.Clone() as NEATNetwork;
                agent.model.Mutate();
            }     
        }


        // NEAT
        private void Evolution()
        {
           Speciate();
           MassKill();
           RemoveEndangeredSpecies();
           Reproduce();

           mainModel.SetFrom(GetBestModel());
           EditorUtility.SetDirty(mainModel);
           AssetDatabase.SaveAssetIfDirty(mainModel);
        }
        void Speciate()
        {
            foreach (var spec in species)
            {
                spec.ClearClients();
            }

            foreach (var agent in population)
            {
                // If he is a representative of a species
                if (agent.GetSpecies() != null)
                    continue;

                // Else introduce him in a species
                bool joined_species = false;
                foreach (var spc in species)
                {
                    if(spc.TryAdd(agent))
                    {
                        joined_species = true;
                        break;
                    }
                }
                
                // If didn't joined any species, create a new species
                if(!joined_species)
                {
                    species.Add(new Species(agent));
                }
            }

            foreach (var spec in species)
            {
                spec.CalculateScore();
            }

           
        }
        void MassKill()
        {
            foreach (var spec in species)
            {
                spec.Kill(1 - Instance.hp.survivalRate);
            }
        }
        void RemoveEndangeredSpecies()
        {
            List<Species> toRemoveSpecies = new List<Species>();
            for (int i = 0; i < species.Count; i++)
            {
                Species spec = species.ElementAt(i);
                if(spec.IsEndangered())
                {
                    spec.GoExtinct();
                    toRemoveSpecies.Add(spec);
                }
            }
            foreach (var toRem in toRemoveSpecies)
            {
                species.Remove(toRem);
            }
            
        }
        void Reproduce()
        {         
            foreach (var agent in population)
            {
                if(agent.GetSpecies() == null)
                {
                    Species spec = GetRandomSpecies();// Species with good overall fitness have more chances to reproduce
                    agent.model = spec.Breed();
                    agent.model.Mutate();
                    spec.ForceAdd(agent);
                }
            }
        }
        void TestAll()
        {
            double[] inputs = new double[population.ElementAt(0).model.inputNodes_cache.Count];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = Functions.RandomValue();
            }

            foreach (var agent in population)
            {                    
                try
                {
                    agent.model.GetDiscreteActions(inputs);
                }
                catch
                {
                    Functions.DebuggerLog(agent.model.ToString());
                    Debug.LogError("err");
                  
                }       
            }
        }

        // Distance
        public static bool AreCompatible(NEATNetwork parent1, NEATNetwork parent2)
        {
            float N = Max_GenesNumber(parent1, parent2);
            float E = Count_ExcessJoints(parent1, parent2);
            float D = Count_Disjoints(parent1, parent2);
            float W = Avg_WeightDifference(parent1, parent2);

            float distance = (Instance.hp.c1 * E / N) +
                             (Instance.hp.c2 * D / N) +
                             (Instance.hp.c3 * W);
            //Functions.DebuggerLog("Distance: " + distance + " |N=" + N + " |E=" + E + " |D=" + D + " |W=" + W);

            return distance < Instance.hp.delta;
        }
        static int Max_GenesNumber(NEATNetwork genome1, NEATNetwork genome2)
        {
            int gen1_count = genome1.connections.Count + genome1.nodes.Count;
            int gen2_count = genome2.connections.Count + genome2.nodes.Count;
            return Mathf.Max(gen1_count, gen2_count);
        }
        static float Avg_WeightDifference(NEATNetwork genome1, NEATNetwork genome2)
        {
            float dif = 0f;
            int matchesCount = 0;

            foreach (var conn1 in genome1.connections)
            {
                foreach (var conn2 in genome2.connections)
                {
                    if (conn1.Key == conn2.Key)
                    {
                        dif += Mathf.Abs(conn1.Value.weight - conn2.Value.weight);
                        matchesCount++;
                    }
                }
            }

            return dif;
        }
        static int Count_ExcessJoints(NEATNetwork genome1, NEATNetwork genome2)
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
        static int Count_Disjoints(NEATNetwork genome1, NEATNetwork genome2)
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

        private string GetEpisodeStatistic()
        {
            StringBuilder text = new StringBuilder();
            text.Append("<color=#2873eb><b>Generation: ");
            text.Append(++GENERATION);
            text.Append(" (");
            text.Append(species.Count);
            text.Append(" species)");
            text.Append("</b></color>\n");
            int index = 1;
            foreach (var spec in species)
            {
                Color color = new Color(FunctionsF.RandomValue(), FunctionsF.RandomValue(), FunctionsF.RandomValue());
                text.Append("<color=");
                text.Append(Functions.HexOf(color));
                text.Append(">\t    Species: ");
                text.Append(index);
                text.Append(" | Members: ");
                text.Append(spec.GetClients().Count);
                text.Append(" | Score: ");
                text.Append(spec.GetScore());
                text.Append("</color>\n");
                index++;
            }
            return text.ToString();
        }
        private Species GetRandomSpecies()
        {
            float min = species.Select(x => x.GetScore()).Min();
            float max = species.Select(x => x.GetScore()).Max();
            float total = max - min;
            float cummulated = min;
            float try_reach = min + FunctionsF.RandomValue() * total;

            foreach (var spec in species)
            {
                cummulated += spec.GetScore();
                if (cummulated > try_reach)
                    return spec;

            }
            // because there might be negative scores, this alg breaks and choose random
            return Functions.RandomIn(species); 
        }
        private NEATNetwork GetBestModel()
        {
            List<NEATAgent> best_foreach_species = species.Select(x => x.GetBestAgent()).ToList();
            NEATAgent best = null;
            foreach (var some in best_foreach_species)
            {
                if (best == null || some.GetFitness() > best.GetFitness())
                    best = some;
            }
            return best.model;
        }

        public static void Dispose() { Destroy(Instance.gameObject); Instance = null; }
        public static int GetInnovation() => Instance.innovationCounter.GetInnovation();
        public static void InitializeHyperParameters() => Instance.hp = new NEATHyperParameters();
        public static NEATHyperParameters GetHP() => Instance.hp;
    }
}

