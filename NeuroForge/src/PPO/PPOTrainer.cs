using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.Threading.Tasks;

namespace NeuroForge
{
    // Used later for parallel training
    public sealed class PPOTrainer
    {
        public static PPOTrainer instance;

        public PPOModel model;
        public ExperienceBuffer[] experienceBuffers;
        public (List<double>, List<double>)[] returns_and_advantages;
        public HyperParameters hp;

        public void Activate()
        {
            // find all agents
            Agent[] agents = GameObject.FindObjectsOfType<Agent>();
            experienceBuffers = agents.Select(ag => ag.Memory).ToArray();

            // get the model
            model = agents[0].model;
            hp = agents[0].hp;
        }
        public void UpdatePolicy()
        {
            foreach (var memory in experienceBuffers)
            {

                //

            }
            
            ClearBuffers();
        }

        private (List<double>, List<double>) GAE(ExperienceBuffer Memory)
        {
            List<Sample> playback = Memory.records;

            //returns = discounted rewards
            List<double> returns = new List<double>();
            List<double> advantages = new List<double>();


            //Normalize rewards 
            double mean = playback.Average(t => t.reward);
            double std = Math.Sqrt(playback.Sum(t => (t.reward - mean) * (t.reward - mean) / playback.Count));
            if (std == 0) std = +1e-8;
            IEnumerable<double> normRewards = playback.Select(r => (r.reward - mean) / std);


            //Calculate returns and advantages
            double advantage = 0;
            for (int i = playback.Count - 1; i >= 0; i--)
            {
                double value = model.criticNetwork.ForwardPropagation(playback[i].state)[0];
                double nextValue = i == playback.Count - 1 ?
                                        0 :
                                        model.criticNetwork.ForwardPropagation(playback[i + 1].state)[0];


                double delta = playback[i].done ?
                               normRewards.ElementAt(i) - value :
                               normRewards.ElementAt(i) + hp.discountFactor * nextValue - value;

                advantage = advantage * hp.discountFactor * hp.gaeFactor + delta;

                // Add normalized advantage
                advantages.Add(model.advantagesNormalizer.Normalize(advantage));
                returns.Add(advantage + value);
            }

            advantages.Reverse();
            returns.Reverse();

            return (returns, advantages);
        }


        private void ClearBuffers()
        {
            for (int i = 0; i < experienceBuffers.Length; i++)
            {
                experienceBuffers[i].Clear();
            }
        }

    }
}