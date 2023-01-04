using System.Linq;
using System.Text;
using UnityEngine;

namespace SmartAgents
{
    public class ActionBuffer : IClearable
    {
        public double[] actions;
        public ActionBuffer(int capacity)
        {
            actions = new double[capacity];
        }
        public ActionBuffer(double[] actions)
        {
            this.actions = actions;
        }

        public void Set(int actionIndex, float actionValue)
        {
            actions[actionIndex] = actionValue;
        }
        public float Get(int actionIndex)
        {
            return (float)actions[actionIndex];
        }


        /// <summary>
        /// Get the index of discrete action (assuming that you already labeled possible actions with integers from 0 to n-1).
        /// </summary>
        /// <returns>[int] index</returns>
        public int RequestDiscreteAction()
        {
            double max = double.MinValue;
            int index = -1;
            bool equal = true;
            for (int i = 0; i < actions.Length; i++)
            {
                if (i > 0 && actions[i] != actions[i - 1])
                    equal = false;

                if (actions[i] > max)
                {
                    max = actions[i];
                    index = i;
                }
            }
            return equal == true ? -1 : index;

        }
        public void Clear()
        {
            actions = Enumerable.Repeat(0.0, actions.Length).ToArray();
        }
        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append("[ ");

            foreach (var obs in actions)
            {
                stringBuilder.Append(obs);
                stringBuilder.Append(", ");
            }
            stringBuilder.Remove(stringBuilder.Length - 2, 1);
            stringBuilder.Append("]");
            return stringBuilder.ToString();
        }
    }

}