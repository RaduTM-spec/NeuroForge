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
        private List<NEATAgent> clients = new List<NEATAgent>();
        private NEATAgent representative;
        private float score = 0;
        public Species(NEATAgent representative)
        {
            this.representative = representative;
            this.clients.Add(representative);
        }        

        public void CalculateScore() => score = clients.Select(x => x.GetFitness()).Average();
        
        public bool TryAdd(NEATAgent agent)
        {
            if(NEATTrainer.AreCompatible(agent.model, representative.model))
            {
                clients.Add(agent);
                agent.SetSpecies(this);
                return true;
            }
            return false;
        }
        public void ForceAdd(NEATAgent agent)
        {
            clients.Add(agent);
            agent.SetSpecies(this);
        }
        public void ClearClients()
        {
            foreach (var cl in clients)
            {
                    cl.SetSpecies(null);
            }
            clients.Clear();

            // Keep the representative
            representative.SetSpecies(this);
            clients.Add(representative);
            
            score = 0;
            
        }
        public void Kill(float percentage01)
        {
            if (clients.Count < 3) return;
            clients.Sort((x, y) => x.GetFitness().CompareTo(y.GetFitness()));
            for (int i = 0; i < clients.Count * percentage01; i++) 
            {
                clients[0].SetSpecies(null);
                clients[0].model = null;
                clients.RemoveAt(0);
            }
            // It is possible to kill the representative in this process, so choose the best client as representative
            if (clients.Count > 0 && !Functions.IsValueIn(representative, clients))
                representative = clients.Last();
        }
        
        public NEATNetwork Breed()
        {
            // Select two random parents based on their fitness
            // Theoretically the agents are sorted at this point
            NEATAgent parent1 = null;
            NEATAgent parent2 = null;


            float total_fit = clients.Sum(x => x.GetFitness());
            float cumulated = 0f;

            float target = FunctionsF.RandomValue() * total_fit;
            foreach (var item in clients)
            {
                cumulated += item.GetFitness();
                if(cumulated >= target)
                {
                    parent1 = item;
                    break;
                }
            }
            target = FunctionsF.RandomValue() * total_fit;
            cumulated = 0f;
            foreach (var item in clients)
            {
                cumulated += item.GetFitness();
                if (cumulated >= target)
                {
                    parent2 = item;
                    break;
                }
            }

            return CrossOver(parent1.model, parent2.model, parent1.GetFitness(), parent2.GetFitness());
        }
        private static NEATNetwork CrossOver(NEATNetwork parent1, NEATNetwork parent2, float p1_fitness, float p2_fitness)
        {
            float sum = p1_fitness + p2_fitness;
            p1_fitness = p1_fitness / sum;
            p2_fitness = p2_fitness / sum;

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
        public void GoExtinct()
        {
            foreach (var ag in clients)
            {
                ag.SetSpecies(null);
            }

            representative = null;
        }

        public bool IsEndangered() => clients.Count < 2;
        public NEATAgent GetBestAgent()
        {
            NEATAgent best_ag = null;
            foreach (var ag in clients)
            {
                if (best_ag == null || ag.GetFitness() > best_ag.GetFitness())
                    best_ag = ag;
            }
            return best_ag;
        }
        public float GetScore() => score;
        public List<NEATAgent> GetClients() => clients;
    }
}
