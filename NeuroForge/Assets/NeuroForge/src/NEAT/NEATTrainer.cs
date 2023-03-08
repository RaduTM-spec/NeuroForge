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
using static Unity.VisualScripting.LudiqRootObjectEditor;
using static UnityEditor.PlayerSettings;

namespace NeuroForge
{
    public sealed class NEATTrainer : MonoBehaviour
    {
        private static NEATTrainer Instance;

        [SerializeField] private List<NEATAgent> population;
        private List<Species> species;

        private Genome mainModel;
        private NEATHyperParameters hp;
        private TransformReseter trainingEnvironment;

        [SerializeField] private int agentsDead = 0;
        [SerializeField] private float episodeTimePassed = 0;

        [SerializeField] private int generation = 0;
        [SerializeField] private bool sessionEnd = false;

        private int speciesID_counter = 0;
        private float fitnessRecord = float.MinValue;

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
                Instance.NEAT_Algorithm2();

                // Reset Environment
                Instance.trainingEnvironment.Reset();

                // Reset Episode Stats
                Instance.agentsDead = 0;
                Instance.episodeTimePassed = 0;

                // Print Episode Statistic
                PrintEpisodeStatistic();

                // Resurrect agents
                foreach (var agent in Instance.population)
                {
                    agent.Resurrect();
                    agent.SetFitness(0f);
                }

                // Check for stop
                if (generation == hp.generations)
                {
                    sessionEnd = true;
                    species = null;
                    foreach (var ag in population)
                    {
                        ag.behavior = BehaviourType.Inactive;
                    }
                    Debug.Log("<color=#2873eb><b>Training session ended!\n\n\n\n\n\n\n\n\n\n\n</b></color>");
                    EditorApplication.isPlaying = false;

                }
            }
        }
        private void OnDrawGizmos()
        {
            if (!mainModel) return;

            bool drawSphereNodes = Instance.hp.nodeShape == NodesDrawShape.Sphere ? true : false;

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
                        Gizmos.color = hp.inputNodesColor;
                        break;
                    case NEATNodeType.output:
                        Gizmos.color = hp.outputNodesColor;
                        break;
                    case NEATNodeType.hidden:
                        Gizmos.color = hp.hiddenNodesColor;
                        break;
                    case NEATNodeType.bias:
                        Gizmos.color = hp.biasNodeColor;
                        break;
                }
                if (drawSphereNodes)
                    Gizmos.DrawSphere(new Vector3(node.Value.x, node.Value.y, node.Value.z), NODE_SCALE);
                else
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

                Vector3 left_right_offset = new Vector3(drawSphereNodes ? NODE_SCALE : 0.5f * NODE_SCALE, 0, 0);
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
        public static void InitializeTrainer(NEATAgent agent)
        {
            if (Instance != null)
                return;

            GameObject go = new GameObject("NEATTrainer");
            go.AddComponent<NEATTrainer>();

            Instance.population = new List<NEATAgent>() { agent };
            Instance.species = new List<Species>();

            Instance.mainModel = agent.model;
            Instance.hp = agent.hp;
            Instance.hp.generations = agent.hp.generations;
            Instance.hp.timeHorizon = agent.hp.timeHorizon;

            if (Instance.hp.OnlySigmoid)
                Instance.hp.mutateNode = 0;

            Instance.InitializeAgents(agent.gameObject, agent.hp.populationSize - 1);
            Instance.trainingEnvironment = new TransformReseter(agent.transform.parent); // is ok placed here, to get reference of all other agents
        }
        private void InitializeAgents(GameObject modelAgent, int size)
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


        // NEAT 1 (original implementation -- actually works extremly bad)
        private void NEAT_Algorithm1()
        {
            //1
            Speciate();
            //2 
            SpeciesCalculations();
            //3
            SpeciesOffspringCalculations();
            //4
            KillPopulation();
            //5
            KillSpeciesWith0Team();
            //6
            Breed();

            // Increment the age foreach species
            foreach (var item in species)
            {
                item.age++;
            }

        }
        private void SpeciesOffspringCalculations()
        {
            float total_sh_fitness = species.Sum(s => s.GetSpeciesSharedFitness());
            int offsprings_assigned = 0;
            foreach (var spec in species)
            {
                int i_offer_this_number = (int)(spec.GetSpeciesSharedFitness() / total_sh_fitness * population.Count);
                spec.no_offsprings_assigned = i_offer_this_number;
                offsprings_assigned += i_offer_this_number;
            }
            // Because the sum of assigned offsprings != population, we give more to random species until we fill up
            for (int i = 0; i < population.Count - offsprings_assigned; i++)
            {
                Species spec = Functions.RandomIn(species);
                spec.no_offsprings_assigned++;
            }
            
        }
        private void KillPopulation()
        {

            // Kill the worst agents (only their genome ofc) from the population (we do not care about their species)
            // Actually this idea is extremly bad, so i changed it
            population.Sort((x, y) => x.GetFitness().CompareTo(y.GetFitness()));
            int how_many_to_kill = (int) (1f - Instance.hp.survivalRate) * population.Count;
            for (int i = 0; i < how_many_to_kill; i++)
            {
                NEATAgent x = population[i];
                x.GetSpecies().TryRemove(x);
                x.SetSpecies(null);
                x.SetFitness(0);
                x.SetAdjustedFitness(0);         
                x.model = null;
            }
        }
        private void KillSpeciesWith0Team()
        {
            List<Species> toRemove = new List<Species>();
            foreach (var spec in species)
            {
                if (spec.GetIndividuals().Count == 0)
                {
                    toRemove.Add(spec);
                }
            }
            foreach (var extSpecies in toRemove)
            {
                extSpecies.GoExtinct();
                species.Remove(extSpecies);
            }
        }
        private void Breed()
        {
            // Keep all offsprings here foreach species
            Dictionary<Species, List<Genome>> offsprings = new Dictionary<Species, List<Genome>>();
            foreach (var spec in species)
            {
                int num = spec.no_offsprings_assigned;
                List<Genome> offs = new List<Genome>();
                for (int i = 0; i < num; i++)
                {
                    offs.Add(spec.Breed());
                }
                offsprings.Add(spec, offs);
            }

            // Clean up old generation
            foreach (var spec in species)
            {
                spec.FullReset();
            }
            foreach (var ind in population)
            {
                ind.model = null;
            }

            // Fill up the species
            foreach (var item in offsprings)
            {
                Species spec = item.Key;
                List<Genome> offs = item.Value;
                foreach (var brain in offs)
                {
                    NEATAgent brainlessAg = GetBrainlessAgent();
                    brainlessAg.model = brain;
                    spec.ForceAdd(brainlessAg);
                }

            }
        }

        // NEAT 2
        private void NEAT_Algorithm2()
        {
            /*
             *  Every species is
             *  assigned a potentially different number of offspring in proportion to the sum of adjusted
             *  fitnesses f0
             *  i of its member organisms. Species then reproduce by first eliminating
             *  the lowest performing members from the population. The entire population is then
             *  replaced by the offspring of the remaining organisms in each species
             */

            Speciate();
            SpeciesCalculations();         
            Culling();
            SpeciesRemoveExtinct();
            Reproduce();
           

            mainModel.SetFrom(GetBestModel());
            EditorUtility.SetDirty(mainModel);
            AssetDatabase.SaveAssetIfDirty(mainModel);
        }
        void Speciate()
        {
            // Reset species - rand indiv becomes the representative
            foreach (var spec in species)
            {
                spec.Reset();
            }

            // Set species
            foreach (var agent in population)
            {
                // Representatives already joined species
                if (agent.GetSpecies() != null)
                    continue; 

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
                    species.Add(new Species(++speciesID_counter, agent));
                }
            }

        }
        void SpeciesCalculations()
        {
            species.ForEach(x => x.AdjustFitness());
            species.ForEach(x => x.CalculateShFitSum());
            species.ForEach(s => s.UpdateStagnation());

            // The sort is done by comparing their champions' fitnesses
            species.Sort((x, y) =>
            {
                float x_fit = x.GetChampion().GetFitness();
                float y_fit = y.GetChampion().GetFitness();
                if (x_fit == y_fit)
                    return 0;
                return x_fit > y_fit ? -1 : 1;

            });
        }
        void Culling()
        {
            foreach (var spec in species)
            {
                spec.Kill(1f - Instance.hp.survivalRate);
            }         
        }
        void SpeciesRemoveExtinct()
        {
            // Species that doesn't reproduce enough are in danger
 
            List<Species> toRemove = new List<Species>();
            foreach (var spec in species)
            {
                if (spec.GetIndividuals().Count < 2 && spec.age > 3)
                {
                    toRemove.Add(spec);
                }
            }
            foreach (var extSpecies in toRemove)
            {
                extSpecies.GoExtinct();
                species.Remove(extSpecies);
            }
        }
        void Reproduce()
        {         
            foreach (var agent in population)
            {
                if(agent.GetSpecies() == null)
                {
                    Species spec = GetNotSoRandomSpecies();
                    agent.model = spec.Breed(); // is mutated there already
                    spec.ForceAdd(agent);
                }
            }

            // Increment the age foreach species
            foreach (var item in species)
            {
                item.age++;
            }
        }

        // Other
        private void PrintEpisodeStatistic()
        {
            StringBuilder text = new StringBuilder();
            text.Append("<color=#2873eb><b>Generation ");
            text.Append(++generation);
            text.Append("</b> (No. species: ");
            text.Append(species.Count);
            text.Append(")</color>\n");
            text.Append("<color=#099c94>_______________________________________________________________________\n</color>");
            text.Append("<color=#099c94>| SpeciesID | Size | Shared Fitness | Best Fitness | Age | Stagnation |\n</color>"); 
            text.Append("<color=#099c94>-----------------------------------------------------------------------\n</color>");
            foreach (var spec in species)
            {
                Color color = new Color(Mathf.Clamp(FunctionsF.RandomValue(),.5f,1f), Mathf.Clamp(FunctionsF.RandomValue(), .5f, 1f), Mathf.Clamp(FunctionsF.RandomValue(), .5f, 1f));

                text.Append("<color=");
                if (spec.stagnation >= hp.stagnationAllowance)
                    text.Append("red");
                else
                    text.Append(Functions.HexOf(color));
                text.Append(">|");

                string id = ("#" + spec.id).PadLeft(10, ' ');             
                string size = spec.GetIndividuals().Count.ToString().PadLeft(5, ' ');
                string shFitness = spec.GetSpeciesSharedFitness().ToString("0.000").PadLeft(15,' ');
                float bestFit = spec.GetChampion().GetFitness();
                string bestFitness = bestFit.ToString("0.000").PadLeft(13, ' ');
                string age = spec.age.ToString().PadLeft(4, ' ');
                string stallness = spec.stagnation < hp.stagnationAllowance || species.Count == 1? "" : "*";
                stallness = (stallness + spec.stagnation.ToString()).PadLeft(11, ' ');

                text.Append(id);

                text.Append(" |");
                text.Append(size);

                text.Append(" |");
                text.Append(shFitness);

                text.Append(" |");
                text.Append(bestFitness);

                text.Append(" |");
                text.Append(age);

                text.Append(" |");
                text.Append(stallness);

                text.Append(" |</color>\n");

                fitnessRecord = Math.Max(fitnessRecord, bestFit);
            }
            text.Append("<color=#099c94>-----------------------------------------------------------------------\n</color>");

            // Insert fit record
            string fit_record = ("<color=#099c94>(Fitness record: <b>" + fitnessRecord.ToString("0.000") + "</b>)</color>").PadLeft(101, ' ');
            text.Append(fit_record);
            Debug.Log(text.ToString());
        }
        private Species GetNotSoRandomSpecies()
        {
            // Note:
            // Paper: Every species is assigned a potentially different number of offspring in proportion to the sum of adjusted fitnesses f'[i] of its member organisms.
            // Though, instead we assigned a probability of breeding to that species.
            // The number assigned for both situation is the sum of the adjusted fitness of all individuals
            // On numbers in range [100-1000], the probability might work ok

            if (species.Count == 1) return species.First();
            
            List<Species> allowed_to_reproduce_species = species.Where(x => x.IsAllowedToReproduce(hp.stagnationAllowance)).ToList();

            // Papar reference:
            // In rare cases when the fitness of the entire population does not improve for more than 20 generations,
            // only the top two species are allowed to reproduce, refocusing the search into the most promising spaces.
            if (allowed_to_reproduce_species.Count == 0)
            {
                allowed_to_reproduce_species.Add(species[0]);
                allowed_to_reproduce_species.Add(species[1]);
            }

            List<float> probs = allowed_to_reproduce_species.Select(x => x.GetSpeciesSharedFitness()).ToList();

            return Functions.RandomIn(allowed_to_reproduce_species, probs);
        }
        private Genome GetBestModel()
        {
            List<NEATAgent> best_foreach_species = species.Select(x => x.GetChampion()).ToList();
            NEATAgent best = null;
            foreach (var some in best_foreach_species)
            {
                if (best == null || some.GetFitness() > best.GetFitness())
                    best = some;
            }
            return best.model;
        }
        private NEATAgent GetBrainlessAgent()
        {
            foreach (var item in population)
            {
                if (item.model == null)
                    return item;
            }
            return null;
        }

        public static void Dispose() { Destroy(Instance.gameObject); Instance = null; }
        public static void InitializeHyperParameters() => Instance.hp = new NEATHyperParameters();
        public static NEATHyperParameters GetHyperParam() => Instance.hp;
        public static ActivationTypeF GetNodeActivation()
        {
            // Better results with random activations!
            // Modified sigmoid is actually shit idk why
            if (Instance.hp.OnlySigmoid)
            {
                if (Instance.mainModel.actionSpace == ActionType.Continuous)
                    return ActivationTypeF.HyperbolicTangent;
                else
                    return ActivationTypeF.ModifiedSigmoid;
            }
            else
                return (ActivationTypeF)(int)(FunctionsF.RandomValue() * Enum.GetValues(typeof(ActivationTypeF)).Length); 
            
        }
    }

    public enum NodesDrawShape
    {
        Sphere,
        Cube
    }
}

