using JetBrains.Annotations;
using System;
using System.Text;
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
        [SerializeField] public double[] nextAction;

        [Space, SerializeField] public bool terminalState;
        [SerializeField] public double discountedReward;

        public Sample(double[] state, double[] action, double reward, bool terminalState)
        {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = null;
            this.nextAction = null;
            this.terminalState = terminalState;
            this.discountedReward = 0;
        }

        public bool IsComplete()
        {
            if (state == null || state.Length == 0) return false;
            if (action == null || action.Length == 0) return false;
            if (nextState == null || nextState.Length == 0) return false;
            return true;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("s:[ ");
            foreach (var item in state)
            {
                sb.Append(item);
                sb.Append(", ");
            }
            sb.Remove(sb.Length- 2, 1);
            sb.Append("] ");

            sb.Append("a:[ ");
            foreach (var item in action)
            {
                sb.Append(item);
                sb.Append(", ");
            }
            sb.Remove(sb.Length - 2, 1);
            sb.Append("] ");

            sb.Append("r: [ ");
            sb.Append(reward);
            sb.Append(" ]");

            return sb.ToString();

        }
    }
}
