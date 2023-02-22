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
        public bool secondChance = false; // when a species is endangered (it doesn't breed well), it is given a second chance episode to breed (once per life-time)
        public Species(NEATAgent repr, bool sc)
        {
            repr.SetSpecies(this);
            this.representative = repr;         
            this.individuals.Add(repr);
            this.secondChance = sc;
        }        

        public void CalculateAvgFitness() => avgFitness = individuals.Select(x => x.GetFitness()).Average();

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
            // never remove this thing here. otherwise there will be error in the last if
            if (individuals.Count < 3) return;

            individuals.Sort((x, y) => x.GetFitness().CompareTo(y.GetFitness()));

            float range = individuals.Count * percentage01;
            for (int i = 0; i < range - 1; i++) 
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
        public NEATNetwork Breed()
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
        public bool IsEndangered(int endangerZone)
        {
            // It has enough individuals, it is not going to extinct
            if(individuals.Count > endangerZone)
                return false;

            // Yes it is endangered, check the second chance
            if (!secondChance)
                return true;
          
            secondChance = false;
            return false;
        }

        private static NEATNetwork CrossOver(NEATNetwork parent1, NEATNetwork parent2, float p1_fitness, float p2_fitness)
        {
            p1_fitness = Mathf.Exp(p1_fitness);
            p2_fitness = Mathf.Exp(p2_fitness);
            float sum = p1_fitness + p2_fitness;
            p1_fitness /= sum;
            p2_fitness /= sum;

            // Notes: matching genes are taken randomly weighted (by fitness) from the parent with highest fitness
            // Notes: the new child (a.k.a. new topology) is mutated afterwards

            NEATNetwork child = new NEATNetwork(parent1.GetInputsNumber(), parent1.outputShape, parent1.actionSpace, false, false);
            child.nodes.Clear();
            child.connections.Clear();

            int max_innovation = Mathf.Max(parent1.GetHighestInnovation(), parent2.GetHighestInnovation());

            // Insert nodes
            for (int i = 0; i <= max_innovation; i++)
            {
                NodeGene parent1_gene = null;
                NodeGene parent2_gene = null;

                if(parent1.nodes.ContainsKey(i))
                    parent1_gene= parent1.nodes[i];
                if(parent2.nodes.ContainsKey(i))
                    parent2_gene= parent2.nodes[i];

                NodeGene clone;
                if (parent1_gene != null && parent2_gene != null)
                {   
                    if(FunctionsF.RandomValue() < p1_fitness)
                        clone = parent1_gene.Clone() as NodeGene;
                    else
                        clone = parent2_gene.Clone() as NodeGene;
                    clone.incomingConnections.Clear();
                    child.nodes.Add(clone.innovation, clone);
                }
                else if(parent1_gene != null)
                {
                    clone = parent1_gene.Clone() as NodeGene;
                    clone.incomingConnections.Clear();
                    child.nodes.Add(clone.innovation, clone);
                }
                else if(parent2_gene != null)
                {
                    clone = parent2_gene.Clone() as NodeGene;
                    clone.incomingConnections.Clear();
                    child.nodes.Add(clone.innovation, clone);
                }
            }

            // Insert connections
            for (int i = 1; i <= max_innovation; i++)
            {
                ConnectionGene parent1_gene = null;
                ConnectionGene parent2_gene = null;

                if(parent1.connections.ContainsKey(i))
                    parent1_gene = parent1.connections[i];
                if (parent2.connections.ContainsKey(i))
                    parent2_gene = parent2.connections[i];

                if (parent1_gene != null && parent2_gene != null)
                {
                    if (FunctionsF.RandomValue() < p1_fitness)
                        child.connections.Add(parent1_gene.innovation, parent1_gene.Clone() as ConnectionGene);
                    else
                        child.connections.Add(parent2_gene.innovation, parent2_gene.Clone() as ConnectionGene);
                }
                else if (parent1_gene != null)
                    child.connections.Add(parent1_gene.innovation, parent1_gene.Clone() as ConnectionGene);
                else if (parent2_gene != null)
                    child.connections.Add(parent2_gene.innovation, parent2_gene.Clone() as ConnectionGene);
            }

            // Calculate incoming connections foreach node
            foreach (var connection in child.connections)
            {
                child.nodes[connection.Value.outNeuron].incomingConnections.Add(connection.Value.innovation);
            }

            child.Mutate();

            return child;
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
        public List<NEATAgent> GetIndividuals() => individuals;
    }
}
