using JetBrains.Annotations;
using System;
using System.Text;
using UnityEngine;
namespace NeuroForge
{
    [Serializable]
    public class Sample
    {
        [SerializeField] public double[] state;
        [SerializeField] public double[] action;
        
        [SerializeField] public double[] log_probs;
        [SerializeField] public double value;

        [SerializeField] public double reward;
        [SerializeField] public bool done;

        public Sample(double[] state, double[] rawOutput, double reward, double[] log_probs, double value, bool isDone)
        {
            this.state = state;
            this.action = rawOutput;
           
            this.log_probs = log_probs;
            this.value = value;

            this.reward = reward;
            this.done = isDone;
        }

    }
}
