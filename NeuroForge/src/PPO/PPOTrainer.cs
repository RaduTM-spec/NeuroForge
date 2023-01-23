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
        public PPOMemory[] experienceBuffers;
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

        private void ClearBuffers()
        {
            for (int i = 0; i < experienceBuffers.Length; i++)
            {
                experienceBuffers[i].Clear();
            }
        }

    }
}