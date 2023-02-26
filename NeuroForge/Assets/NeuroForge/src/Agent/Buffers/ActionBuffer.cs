using System.Linq;
using System.Text;
using UnityEngine;

namespace NeuroForge
{
    public class ActionBuffer : IClearable
    {
        public float[] ContinuousActions;
        public int[] DiscreteActions;
        public ActionBuffer(int capacity)
        {
            ContinuousActions = new float[capacity];
            DiscreteActions = new int[capacity];
        }

        public double[] ActionsToDouble(ActionType type)
        {

            switch(type)
            {
                case ActionType.Continuous:
                    return ContinuousActions.Select(x => (double)x).ToArray();
                case ActionType.Discrete:
                    return DiscreteActions.Select(x => (double)x).ToArray();
                default:
                    throw new System.Exception("invalid actions type");
            }

        }
        public void Clear()
        {
            ContinuousActions = Enumerable.Repeat(0f, ContinuousActions.Length).ToArray();
            DiscreteActions = Enumerable.Repeat(0, DiscreteActions.Length).ToArray();
        }
        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.Append("continuous:[ ");
            foreach (var c in ContinuousActions)
            {
                stringBuilder.Append(c);
                stringBuilder.Append(", ");
            }
            stringBuilder.Remove(stringBuilder.Length - 2, 1);
            stringBuilder.Append("]");

            stringBuilder.Append("discrete:[ ");
            foreach (var d in DiscreteActions)
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