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
        private List<NEATAgent> agents = new List<NEATAgent>();
        private NEATAgent representative;
        private float SPECIE_SCORE = 0;
        public Species(NEATAgent representative)
        {
            this.representative = representative;
            this.agents.Add(representative);
        }
        ~Species() { }
        

        public float CalculateScore() => SPECIE_SCORE = agents.Select(x => x.GetFitness()).Average();
        
        public bool TryAdd(NEATAgent agent)
        {
            if(NEATTrainer.AreCompatible(agent.model, representative.model))
            {
                agents.Add(agent);
                agent.SetSpecies(this);
                return true;
            }
            return false;
        }
        public void ForceAdd(NEATAgent agent)
        {
            agents.Add(agent);
            agent.SetSpecies(this);
        }
        public void Reset()
        {
            representative = Functions.RandomIn(agents);
            foreach (var cl in agents)
            {
                cl.SetSpecies(null);
            }
            agents.Clear();

            representative.SetSpecies(this);
            agents.Add(representative);
        }
        public void Kill(float percentage01)
        {
            for (int i = 0; i < agents.Count - 1; i++)
            {
                for (int j = i+1; j < agents.Count; j++)
                {
                    if (agents[i].GetFitness() > agents[j].GetFitness())
                    {
                        NEATAgent tmp = agents[i];
                        agents[i] = agents[j];
                        agents[j] = tmp;
                    }
                }              
            }

            for (int i = 0; i < agents.Count * percentage01; i++)
            {
                agents[0].SetSpecies(null);
                agents[0].model = null;
                agents.RemoveAt(0);
            }
            // even if the representative is killed, it still remains the representative
        }
        public void GoExtinct()
        {
            foreach (var ag in agents)
            {
                ag.SetSpecies(null);
                
            }
            representative = null;
        }
        public NEATNetwork Breed()
        {
            NEATAgent parent1 = Functions.RandomIn(agents);
            NEATAgent parent2 = Functions.RandomIn(agents);
            return CrossOver(parent1.model, parent2.model, parent1.GetFitness(), parent2.GetFitness());
        }
        private static NEATNetwork CrossOver(NEATNetwork parent1, NEATNetwork parent2, float p1_fitness, float p2_fitness)
        {
            float sum = p1_fitness + p2_fitness;
            p1_fitness = p1_fitness / sum;
            p2_fitness = p2_fitness / sum;

            // Notes: matching genes are taken randomly weighted (by fitness) from the parent with highest fitness
            // Notes: the new child (a.k.a. new topology) is mutated afterwards

            NEATNetwork child = new NEATNetwork(parent1.GetInputsNumber(), parent1.outputShape, parent1.actionSpace, false);
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

            return child;
        }

        public NEATAgent GetBestAgent()
        {
            NEATAgent best_ag = null;
            foreach (var ag in agents)
            {
                if (best_ag == null || ag.GetFitness() > best_ag.GetFitness())
                    best_ag = ag;
            }
            return best_ag;
        }
        public int GetSize() => agents.Count;
        public float GetScore() => SPECIE_SCORE;
        public List<NEATAgent> GetAgents() => agents;
    }
}
