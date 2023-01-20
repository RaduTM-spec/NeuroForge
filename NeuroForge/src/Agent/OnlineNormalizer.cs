using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    public class OnlineNormalizer
    {
        double[] min;
        double[] max;

        public OnlineNormalizer(int size)
        {
            min = new double[size];
            max = new double[size];
            for (int i = 0; i < size; i++)
            {
                min[i] = double.MaxValue;
                max[i] = double.MinValue;
            }
        }

        private void Update(double[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] < min[i])
                {
                    min[i] = values[i];
                }
                if (values[i] > max[i])
                {
                    max[i] = values[i];
                }
            }
        }
        public void Normalize(double[] tuple)
        {
            Update(tuple);
            for (int i = 0; i < tuple.Length; i++)
            {
                tuple[i] = (tuple[i] - min[i]) / (max[i] - min[i] + 1e-8);
            }
        }
        public double Normalize(double value)
        {
            Update(new double[] {value});
            return (value - min[0]) / (max[0] - min[0] + 1e-8);
        }

    }
}