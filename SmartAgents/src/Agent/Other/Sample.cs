using JetBrains.Annotations;
using System;
using UnityEngine;
namespace SmartAgents
{
    [Serializable]
    public class Sample
    {
        [SerializeField] public double[] state;
        [SerializeField] public double[] action;
        [SerializeField] public double reward;
        [SerializeField] public double[] nextState;

        public Sample(double[] state, double[] action, double reward, double[] nextState)
        {
            this.state= state;
            this.action= action;
            this.reward= reward;
            this.nextState= nextState;
        }
    }
}
