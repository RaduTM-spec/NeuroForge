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
    public sealed class NEATTrainer : MonoBehaviour
    {
        private static NEATTrainer Instance;

        [SerializeField] private HashSet<NEATAgent> population;
        private List<Species> species;

        private Genome mainModel;
        private NEATHyperParameters hp;
        private TransformReseter trainingEnvironment;

        [SerializeField] private int agentsDead = 0;
        [SerializeField] private float episodeTimePassed = 0;

        [SerializeField] private int generation = 0;
        [SerializeField] private bool sessionEnd = false;


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
            if (!sessionEnd && Instance != null && (Instance.episodeTimePassed >= Instance.hp.timeHorizon || Instance.agentsDead == Instance.population.Count))
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
                if(generation == hp.generations)
                {
                    sessionEnd = true;
                    species = null;
                    foreach (var ag in population)
                    {
                        ag.behavior = BehaviourType.Inactive;
                    }
                    Debug.Log("<color=green> Training session ended! </color>");
                    EditorApplication.isPlaying = false;
                   
                }
            }
        }
        private void OnDrawGizmos()
        {
            if (!mainModel) return;

            Dictionary<NodeGene, Vector3> node_pos = new Dictionary<NodeGene, Vector3>();

            Dictionary<float, List<NodeGene>> layer_nodes = new Dictionary<float, List<NodeGene>>();
            foreach (var node in mainModel.nodes.Values)
            {
                if (layer_nodes.ContainsKey(node.layer))
                    layer_nodes[node.layer].Add(node);
                else
                    layer_nodes.Add(node.layer, new List<NodeGene> { node });
            }

            float NODE_SCALE = (4f / mainModel.nodes.Count);
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

            var layers = mainModel.layers;

            for (int i = 1; i < layers.Count - 1; i++)
            {
                var hids_on_this_layer = layer_nodes[layers[i]];
                foreach (var hidden_node in hids_on_this_layer)
                {
                    // get the average Y position of all previous nodes connected with 
                    Dictionary<NodeGene, Vector3> previous_nodes = node_pos.Where(x => x.Key.layer <= layers[i - 1]).ToDictionary(x => x.Key, x => x.Value);

                    List<ConnectionGene> incomingCons = new List<ConnectionGene>();
                    hidden_node.incomingConnections.ForEach(x => incomingCons.Add(mainModel.connections[x]));

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
            foreach (var connection in mainModel.connections)
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
            Instance.species = new List<Species>();

            Instance.mainModel = agent.model;
            Instance.hp = agent.hp;
            Instance.hp.generations = agent.hp.generations;
            Instance.hp.timeHorizon = agent.hp.timeHorizon;

            
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
                agent.model = mainModel.Clone() as Genome;
                agent.model.Mutate();
            }     
        }


        // NEAT
        private void Evolution()
        {
            Speciate();
            SortSpeciesByFitness();         
            Culling();
            Reproduce();
            SpeciesExtinction();

            mainModel.SetFrom(GetBestModel());
            EditorUtility.SetDirty(mainModel);
            AssetDatabase.SaveAssetIfDirty(mainModel);
        }
        void Speciate()
        {
            // Clear species
            foreach (var spec in species)
            {
                spec.ClearClients();
            }

            // Set species
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

            // Increment the age foreach species
            foreach (var item in species)
            {
                item.age++;
            }
        }
        void SortSpeciesByFitness()
        {
            // Calculate score
            foreach (var spec in species)
            {
                spec.CalculateAvgFitness();
            }
            species.Sort((x, y) => x.GetFitness().CompareTo(y.GetFitness()));
        }
        void Culling()
        {
            foreach (var spec in species)
            {
                spec.Kill(1f - Instance.hp.survivalRate);
            }
        }
        void Reproduce()
        {         
            foreach (var agent in population)
            {
                if(agent.GetSpecies() == null)
                {
                    Species spec = GetRandomSpecies_AllowedToReproduce();// Species with good overall fitness have more chances to reproduce
                    agent.model = spec.Breed(); // is mutated there already
                    spec.ForceAdd(agent);
                }
            }
        }
        void SpeciesExtinction()
        {
            // Species that doesn't reproduce this episode and have low individuals are gone
            List<Species> toExtinctSpecies = new List<Species>();
            foreach (var spec in species)
            {
                if(spec.GetIndividuals().Count < 2)
                {
                    toExtinctSpecies.Add(spec);
                }
            }
            foreach (var extinctSpecies in toExtinctSpecies)
            {
                extinctSpecies.GoExtinct();
                species.Remove(extinctSpecies);
            }

            // The killed agents are reproduced
            Reproduce();
        }

        // Distance
        public static bool AreCompatible(Genome parent1, Genome parent2)
        {
            float N = Max_GenesNumber(parent1, parent2);
            float E = Count_ExcessJoints(parent1, parent2);
            float D = Count_Disjoints(parent1, parent2);
            float W = Avg_WeightDifference(parent1, parent2);

            float distance = (Instance.hp.c1 * E / N) +
                             (Instance.hp.c2 * D / N) +
                             (Instance.hp.c3 * W);

            return distance < Instance.hp.delta;
        }
        static int Max_GenesNumber(Genome genome1, Genome genome2)
        {
            int gen1_count = genome1.connections.Count + genome1.nodes.Count;
            int gen2_count = genome2.connections.Count + genome2.nodes.Count;

            int max_gene = Mathf.Max(gen1_count, gen2_count);

            // return Mathf.Clamp(max_gene - 20, 1, max_gene); // max_gene_number is normalized as the paper says
            return max_gene;
        }
        static float Avg_WeightDifference(Genome genome1, Genome genome2)
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
        static int Count_ExcessJoints(Genome genome1, Genome genome2)
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
        static int Count_Disjoints(Genome genome1, Genome genome2)
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
            text.Append(++generation);
            text.Append(" (");
            text.Append(species.Count);
            text.Append(" species)");
            text.Append("</b></color>\n");

            text.Append("<color=#099c94>\t    Species | Age | Size | Fitness</color>\n");
            species.Reverse();
            foreach (var spec in species)
            {
                Color color = new Color(Mathf.Clamp(FunctionsF.RandomValue(),.5f,1f), Mathf.Clamp(FunctionsF.RandomValue(), .5f, 1f), Mathf.Clamp(FunctionsF.RandomValue(), .5f, 1f));

                text.Append("<color=");
                text.Append(Functions.HexOf(color));
                text.Append(">\t        ");

                string id = ("#" + spec.id).PadLeft(7, ' ');
                string age = spec.age.ToString().PadLeft(3, ' ');
                string size = spec.GetIndividuals().Count.ToString().PadLeft(4, ' ');
                string fitness = spec.GetFitness().ToString("G5").PadLeft(6,' ');

                text.Append(id);
                text.Append("      ");
                text.Append(age);
                text.Append("    ");
                text.Append(size);
                text.Append("    ");
                text.Append(fitness);
                text.Append("</color>\n");
            }
            return text.ToString();
        }
        private Species GetRandomSpecies_AllowedToReproduce()
        {
            if (species.Count == 1) return species.First();
            
            List<Species> allowed_to_reproduce_species = species.Where(x => x.IsAllowedToReproduce(hp.stagnationAllowance)).ToList();
            List<float> probs = allowed_to_reproduce_species.Select(x => x.GetFitness()).ToList();

            return Functions.RandomIn(allowed_to_reproduce_species, probs);
        }
        private Genome GetBestModel()
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
        public static void InitializeHyperParameters() => Instance.hp = new NEATHyperParameters();
        public static NEATHyperParameters GetHP() => Instance.hp;
    }
}

