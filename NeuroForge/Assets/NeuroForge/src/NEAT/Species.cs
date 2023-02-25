using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    public class Species
    {
        private List<NEATAgent> individuals = new List<NEATAgent>();
        private NEATAgent representative;
        private float avgFitness = 0;
        private int stagnation = 0;
        // TO DO , the species that do not increase their score in 15 generations, are not allowed to reproduce

        public int id;
        public int age = -1;
        public Species(NEATAgent repr)
        {
            repr.SetSpecies(this);
            this.representative = repr;         
            this.individuals.Add(repr);
            id = UnityEngine.Random.Range(0, 100);
        }        

        public void CalculateAvgFitness()
        {
            float new_average_fitness = individuals.Select(x => x.GetFitness()).Average();
            if (new_average_fitness <= avgFitness)
                stagnation++;
            else
                stagnation = 0;
            avgFitness = new_average_fitness;

        }

        public void ClearClients()
        {
            // Clear everything but representative

            representative = Functions.RandomIn(individuals);

            foreach (var cl in individuals)
            {
                cl.SetSpecies(null);
            }
            individuals.Clear();

            representative.SetSpecies(this);
            individuals.Add(representative);

            avgFitness = 0;

        }
        public bool TryAdd(NEATAgent agent)
        {
            if(NEATTrainer.AreCompatible(agent.model, representative.model))
            {
                individuals.Add(agent);
                agent.SetSpecies(this);
                return true;
            }
            return false;
        }
        public void ForceAdd(NEATAgent agent)
        {
            individuals.Add(agent);
            agent.SetSpecies(this);
        }      
        public void Kill(float percentage01)
        {
            individuals.Sort((x, y) => x.GetFitness().CompareTo(y.GetFitness()));

            float range = individuals.Count * percentage01;
            for (int i = 0; i < range && individuals.Count > 1; i++) 
            {
                individuals[0].SetSpecies(null);
                individuals[0].model = null;
                individuals.RemoveAt(0);
               
            }

            // It is possible to kill the representative in this process, so choose another one
            if (representative.GetSpecies() == null)
            {
                representative = individuals.Last();
            }

            
        }
        public void GoExtinct()
        {
            foreach (var ag in individuals)
            {
                ag.model = null;
                ag.SetSpecies(null);
            }
            individuals.Clear();
            representative = null;
        }
        public Genome Breed()
        {
            // Select two random parents based on their fitness
            // Theoretically the agents are sorted at this point

            NEATAgent parent1 = null;
            NEATAgent parent2 = null;

            List<float> probs = individuals.Select(x => x.GetFitness()).ToList();

            parent1 = Functions.RandomIn(individuals, probs);
            parent2 = Functions.RandomIn(individuals, probs);

            return CrossOver(parent1.model, parent2.model, parent1.GetFitness(), parent2.GetFitness());
        }


        private static Genome CrossOver(Genome parent1, Genome parent2, float p1_fitness, float p2_fitness)
        {
            // parent1 is set as the fittest parent
            if (p1_fitness < p2_fitness)
                Functions.Swap(ref parent1, ref parent2);

            // Notes: matching genes are taken randomly 50/50
            //        disjoint or excess genes are taken from the fittest parent
            //        the offspring is mutated afterwards

            Genome offspring = new Genome(parent1.GetInputsNumber(), parent1.outputShape, parent1.actionSpace, false, false);
            offspring.nodes.Clear();
            offspring.connections.Clear();


            // Insert nodes
            int max_node_id = Math.Max(parent1.GetLastNodeId(), parent2.GetLastNodeId());
            for (int i = 1; i <= max_node_id; i++)
            {
                NodeGene parent1_node = null;
                NodeGene parent2_node = null;

                if(parent1.nodes.ContainsKey(i))
                    parent1_node= parent1.nodes[i];
                if(parent2.nodes.ContainsKey(i))
                    parent2_node= parent2.nodes[i];

                NodeGene clone;
                if (parent1_node != null && parent2_node != null)
                {   
                    if(FunctionsF.RandomValue() < 0.5f)
                        clone = parent1_node.Clone() as NodeGene;
                    else
                        clone = parent2_node.Clone() as NodeGene;

                    clone.incomingConnections.Clear();
                    offspring.nodes.Add(clone.id, clone);
                }
                else if(parent1_node != null)
                {
                    clone = parent1_node.Clone() as NodeGene;
                    clone.incomingConnections.Clear();
                    offspring.nodes.Add(clone.id, clone);
                }
              
            }

            // Insert connections
            int max_conn_innovation = Mathf.Max(parent1.GetLastInnovation(), parent2.GetLastInnovation());         
            for (int i = 1; i <= max_conn_innovation; i++)
            {
                ConnectionGene parent1_gene = null;
                ConnectionGene parent2_gene = null;

                if(parent1.connections.ContainsKey(i))
                    parent1_gene = parent1.connections[i];
                if (parent2.connections.ContainsKey(i))
                    parent2_gene = parent2.connections[i];

                if (parent1_gene != null && parent2_gene != null)
                {
                    ConnectionGene clone = FunctionsF.RandomValue() < 0.5f ?
                            parent1_gene.Clone() as ConnectionGene :
                            parent2_gene.Clone() as ConnectionGene;

                    // if the gene is disabled in both parents, 75% chances of inherited gene to be disabled. Otherwise if enabled by default
                    if (!parent1_gene.enabled && !parent2_gene.enabled)
                        clone.enabled = FunctionsF.RandomValue() < 0.75f ? false : true;
                    else
                        clone.enabled = true;

                    offspring.connections.Add(clone.innovation, clone);
                }
                else if (parent1_gene != null)
                {
                    var clone = parent1_gene.Clone() as ConnectionGene;
                    clone.enabled = true;
                    offspring.connections.Add(clone.innovation, clone);
                }
            }

            // Calculate incoming connections foreach node
            foreach (var connection in offspring.connections)
            {
                offspring.nodes[connection.Value.outNeuron].incomingConnections.Add(connection.Value.innovation);
            }

            // Offspring receives similar layers with fittest parent
            offspring.layers = parent1.layers.ToList();

            offspring.Mutate();
            return offspring;
        }
        

        public NEATAgent GetBestAgent()
        {
            NEATAgent best_ag = null;
            foreach (var ag in individuals)
            {
                if (best_ag == null || ag.GetFitness() > best_ag.GetFitness())
                    best_ag = ag;
            }
            return best_ag;
        }
        public float GetFitness() => avgFitness;
        public bool IsAllowedToReproduce(int stagnationAllowance) => stagnationAllowance > stagnation;
        public List<NEATAgent> GetIndividuals() => individuals;
    }
}
