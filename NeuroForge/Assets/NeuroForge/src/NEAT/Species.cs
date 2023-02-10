using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NeuroForge
{
    public class Species
    {
        private List<NEATAgent> agents = new List<NEATAgent>();
        private NEATAgent representative;

        public Species(NEATAgent representative)
        {
            this.representative = representative;
            this.agents.Add(representative);
        }
        public static NEATNetwork CrossOver(NEATNetwork parent1, NEATNetwork parent2, float p1_fitness, float p2_fitness)
        {

            // Notes: disjoint and excess are taken from the parent with highest fitness
            // Notes: the new child (a.k.a. new topology) is mutated afterwards

            NEATNetwork child = new NEATNetwork(parent1.GetInputsNumber(), parent1.outputShape, parent1.actionSpace, false);


            return child;
        }

        public float GetAverageFitness()
        {
            return agents.Select(x => x.GetFitness()).Average();
        }
        public bool Add(NEATAgent agent)
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
            agents.Sort((x, y) =>
            {
                if (x.GetFitness() > y.GetFitness())
                    return 1;
                else
                    return -1;
            });
            for (int i = 0; i < agents.Count * percentage01; i++)
            {
                agents[i].SetSpecies(null);
                agents.RemoveAt(0);
            }
            // even if the representative is killed, it still remains the representative
        }
        public NEATNetwork Breed()
        {
            NEATAgent parent1 = Functions.RandomIn(agents);
            NEATAgent parent2 = Functions.RandomIn(agents);
            return CrossOver(parent1.model, parent2.model, parent1.GetFitness(), parent2.GetFitness());
        }
    }
}
