using Palmmedia.ReportGenerator.Core;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    public class Species : IResetable
    {       
        public int id;
        private float bestFitness = float.MinValue; // best fitness managed by an individual ever in this species (not shared)
        private float sharedFitnessSum = float.MinValue;
        public int stagnation = 0;
        public int age = -1;
        public int no_offsprings_assigned;

        private List<NEATAgent> individuals = new List<NEATAgent>();
        private NEATAgent representative;

        public Species(int id, NEATAgent repr)
        {
            repr.SetSpecies(this);
            this.representative = repr;         
            this.individuals.Add(repr);
            this.id = id;
        }        
        public void Reset()
        {
            // Paper reference: 
            // Each existing species is represented by a random genome inside the species from the previous generation.

            representative = Functions.RandomIn(individuals);

            foreach (var ind in individuals)
            {
                ind.SetSpecies(null);
            }
            individuals.Clear();
            
            representative.SetSpecies(this);
            individuals.Add(representative);
        }
        public void FullReset()
        {
            foreach (var item in individuals)
            {
                item.SetSpecies(null);
            }
            individuals.Clear();

            representative = null;
        }


        // Joining/Exiting
        public bool TryRemove(NEATAgent agent)
        {
            return individuals.Remove(agent);
        }
        public bool TryJoin(NEATAgent agent)
        {
            if(AreCompatible(agent.model, representative.model))
            {
                individuals.Add(agent);
                agent.SetSpecies(this);
                return true;
            }
            return false;
        }
        public void Join(NEATAgent agent)
        {
            individuals.Add(agent);
            agent.SetSpecies(this);
        }      

        // Killing
        public void Kill(float percentage01)
        {
            individuals.Sort((x, y) => x.GetFitness().CompareTo(y.GetFitness()));
            float range = individuals.Count * percentage01;
            for (int i = 0; i < range && individuals.Count > 1; i++) 
            {
                individuals[0].SetFitness(0);
                individuals[0].SetAdjustedFitness(0);
                individuals[0].SetSpecies(null);
                individuals[0].model = null;
                individuals.RemoveAt(0);
               
            }

            // It is possible to kill the representative in this process, so choose another one randomly, even if we do not really care at this moment
            if (representative.GetSpecies() == null)
            {
                representative = Functions.RandomIn(individuals);
            }           
        }
        public void GoExtinct()
        {
            foreach (var ag in individuals)
            {
                ag.model = null;
                ag.SetSpecies(null);
            }
            representative = null;
        }
        public void AdjustFitness()
        {
            // f'[i] = f[i] / Sum(sh(i,j))
            // f' = shared fitness
            // Sum is the sum of total fitnesses in this species
            // sh is 1 if (i,j) are compatible (they do not exceed delta)
            // sh is 0 if (i,j) are not compatible

            foreach (var indiv in individuals)
            {
                indiv.SetAdjustedFitness(indiv.GetFitness() / individuals.Count);
            }
        }

        // Breeding
        public Genome Breed()
        {
            Genome offspring = null;
            List<float> probs = individuals.Select(x => x.GetFitness()).ToList();//Here works with normal fit too

            // 25% asexual breeding
            if (FunctionsF.RandomValue() < NEATTrainer.GetHyperParam().cloneBreeding)
            {
                
                NEATAgent parent = Functions.RandomIn(individuals, probs);  
                
                offspring = parent.model.Clone() as Genome;
            }
            // 75% crossover breeding
            else
            {
                NEATAgent parent1 = Functions.RandomIn(individuals, probs);
                NEATAgent parent2 = Functions.RandomIn(individuals, probs);

                offspring = CrossOver(parent1.model, parent2.model, parent1.GetFitness(), parent2.GetFitness());
            }

            offspring.Mutate();
            return offspring;
        }    
        private static Genome CrossOver(Genome parent1, Genome parent2, float p1_fitness, float p2_fitness)
        {
            // Parent1 is set as the fittest parent
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
                    if(Functions.RandomValue() < 0.5)
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
            int max_conn_innovation = Math.Max(parent1.GetLastInnovation(), parent2.GetLastInnovation());         
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
                    ConnectionGene clone = Functions.RandomValue() < 0.5 ?
                            parent1_gene.Clone() as ConnectionGene :
                            parent2_gene.Clone() as ConnectionGene;

                    // if the gene is disabled in both parents, 75% chances of inherited gene to be disabled. Otherwise if enabled by default
                    if (!parent1_gene.enabled && !parent2_gene.enabled)
                        clone.enabled = Functions.RandomValue() < 0.75 ? false : true;
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
          
            return offspring;
        }

        // Similarity
        static bool AreCompatible(Genome genome1, Genome genome2)
        {
            float delta = NEATTrainer.GetHyperParam().delta;
            float c1 = NEATTrainer.GetHyperParam().c1;
            float c2 = NEATTrainer.GetHyperParam().c2;
            float c3 = NEATTrainer.GetHyperParam().c3;

            float N = Calculate_N(genome1, genome2);
            float E = Calculate_E(genome1, genome2);
            float D = Calculate_D(genome1, genome2);
            float W = Calculate_W(genome1, genome2);

            float distance = (c1 * E / N) + (c2 * D / N) + (c3 * W);
            return distance < delta;
        }
        static int Calculate_N(Genome genome1, Genome genome2)
        {
            // N is the length of the largest genome
            int N = Math.Max(genome1.connections.Count, genome2.connections.Count);

            // Normalize N (as in original paper)
            N = Math.Max(1, N - 20);

            return N;
        }     
        static int Calculate_E(Genome genome1, Genome genome2)
        {
            int excessJoints = 0;
            int highestMatch = 0;

            // Find highest match, excess joints are higher than this number
            foreach (var conn1 in genome1.connections.Keys)
            {
                foreach (var conn2 in genome2.connections.Keys)
                {
                    if (conn1 == conn2)
                    {
                        highestMatch = Math.Max(highestMatch, conn1);
                        break;
                    }
                }
            }

            foreach (var conn in genome1.connections.Keys)
            {
                if (conn > highestMatch)
                    excessJoints++;
            }
            foreach (var conn in genome2.connections.Keys)
            {
                if (conn > highestMatch)
                    excessJoints++;
            }

            return excessJoints;
        }
        static int Calculate_D(Genome genome1, Genome genome2)
        {
            int disJoints = 0;
            int highestMatch = 0;

            // Calculate highest match, disjoints are less than this
            foreach (var conn1 in genome1.connections.Keys)
            {
                foreach (var conn2 in genome2.connections.Keys)
                {
                    if (conn1 == conn2)
                    {
                        highestMatch = Math.Max(highestMatch, conn1);
                        break;
                    }
                }
            }

            // now check for disjoints (need to be less than the highest match)
            foreach (var conn1 in genome1.connections.Keys)
            {
                bool isMatch = false;
                foreach (var conn2 in genome2.connections.Keys)
                {
                    if (conn1 == conn2)
                    {
                        isMatch = true;
                        break;
                    }
                }
                if (!isMatch && conn1 < highestMatch)
                    disJoints++;
            }
            foreach (var conn2 in genome2.connections.Keys)
            {
                bool isMatch = false;
                foreach (var conn1 in genome1.connections.Keys)
                {
                    if (conn2 == conn1)
                    {
                        isMatch = true;
                        break;
                    }
                }
                if (!isMatch && conn2 < highestMatch)
                    disJoints++;
            }

            return disJoints;
        }
        static float Calculate_W(Genome genome1, Genome genome2)
        {
            if (genome1.connections.Count == 0 && genome2.connections.Count == 0)
                return 0;

            float dif = 1e-8f;
            float matchesCount = 1e-10f;

            foreach (var conn1 in genome1.connections)
            {
                foreach (var conn2 in genome2.connections)
                {
                    if (conn1.Key == conn2.Key)
                    {
                        dif += Mathf.Abs(conn1.Value.weight - conn2.Value.weight);
                        matchesCount++;
                        break;
                    }
                }
            }
            
            return dif / matchesCount;
        }


        // Other
        public void CalculateShFitSum() => sharedFitnessSum = individuals.Sum(x => x.GetAdjustedFitness());
        public float GetSpeciesSharedFitness() => sharedFitnessSum;
        public void UpdateStagnation()
        {
            NEATAgent bestAgent = GetChampion();

            if (bestAgent.GetFitness() <= bestFitness)
                stagnation++;
            else
                stagnation = 0;

            bestFitness = Math.Max(bestFitness, bestAgent.GetFitness());
        }
        public bool IsAllowedToReproduce(int stagnationAllowance)
        {
            if (stagnationAllowance == 0)
                return true;

            return stagnationAllowance > stagnation;
        }
        public NEATAgent GetChampion()
        {
            NEATAgent best_ag = null;
            foreach (var ag in individuals)
            {
                if (best_ag == null || ag.GetFitness() > best_ag.GetFitness())
                    best_ag = ag;
            }
            return best_ag;
        }
        public List<NEATAgent> GetIndividuals() => individuals;
    }
}
