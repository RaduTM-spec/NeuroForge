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
        [SerializeField] public double advantage;
        [SerializeField] public bool done;

        public Sample(double[] state, double[] action, double reward, bool isDone)
        {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.done = isDone;
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
