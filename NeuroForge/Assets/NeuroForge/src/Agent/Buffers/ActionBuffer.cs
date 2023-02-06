using System.Linq;
using System.Text;
using UnityEngine;

namespace NeuroForge
{
    public class ActionBuffer : IClearable
    {
        public float[] continuousActions;
        public int[] discreteActions;
        public ActionBuffer(int capacity)
        {
            continuousActions = new float[capacity];
            discreteActions = new int[capacity];
        }

        public double[] ActionsToDouble(ActionType type)
        {

            switch(type)
            {
                case ActionType.Continuous:
                    return continuousActions.Select(x => (double)x).ToArray();
                case ActionType.Discrete:
                    return discreteActions.Select(x => (double)x).ToArray();
                default:
                    throw new System.Exception("invalid actions type");
            }

        }
        public void Clear()
        {
            continuousActions = Enumerable.Repeat(0f, continuousActions.Length).ToArray();
            discreteActions = Enumerable.Repeat(0, discreteActions.Length).ToArray();
        }
        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.Append("continuous:[ ");
            foreach (var c in continuousActions)
            {
                stringBuilder.Append(c);
                stringBuilder.Append(", ");
            }
            stringBuilder.Remove(stringBuilder.Length - 2, 1);
            stringBuilder.Append("]");

            stringBuilder.Append("discrete:[ ");
            foreach (var d in discreteActions)
            {
                stringBuilder.Append(d);
                stringBuilder.Append(", ");
            }
            stringBuilder.Remove(stringBuilder.Length - 2, 1);
            stringBuilder.Append("]");

            return stringBuilder.ToString();
        }
    }

}