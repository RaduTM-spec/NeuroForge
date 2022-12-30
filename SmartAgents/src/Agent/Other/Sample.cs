using JetBrains.Annotations;
using System;
using UnityEngine;
namespace SmartAgents
{
    [Serializable]
    public struct Sample
    {
        [SerializeField] public double[] state;
        [SerializeField] public double[] action;
        [SerializeField] public double reward;
        [SerializeField] public double[] nextState;
        [SerializeField] public double[] nextAction;

        public Sample(double[] state, double[] action, double reward, double[] nextState, double[] nextAction)
        {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.nextAction = nextAction;
        }

        public bool IsComplete()
        {
            if (state == null || state.Length == 0) return false;
            if (action == null || action.Length == 0) return false;
            if (nextState == null || nextState.Length == 0) return false;
            if (nextAction == null || nextAction.Length == 0) return false;
            return true;
        }
    }
}
