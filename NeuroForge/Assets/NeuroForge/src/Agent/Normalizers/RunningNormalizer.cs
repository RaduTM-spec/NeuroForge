using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class RunningNormalizer
    {
        [SerializeField] double[] min;
        [SerializeField] double[] max;

        public static void OfflineNormalize01(List<double> list, Func<double, double> func)
        {
            double min = list.Min(func);
            double max = list.Max(func);
            for (int i = 0; i < list.Count; i++)
            {
                list[i] = (list[i] - min) / (max - min);
            }

        }
        public static void OfflineNormalizeMinusOneOne(List<double> list, Func<double, double> func)
        {
            double min = list.Min(func);
            double max = list.Max(func);
            for (int i = 0; i < list.Count; i++)
            {
                list[i] = 2 * (list[i] - min) / (max - min) - 1;
            }

        }

        public RunningNormalizer(int size)
        {
             min = new double[size];
             max = new double[size];
             for (int i = 0; i < size; i++)
             {
                 min[i] = double.MaxValue;
                 max[i] = double.MinValue;
             }
            
        }

        public void OptimizeNormalizer(double[] tuple)
        {
            for (int i = 0; i < tuple.Length; i++)
            {
                if (tuple[i] < min[i])
                {
                    min[i] = tuple[i];
                }
                if (tuple[i] > max[i])
                {
                    max[i] = tuple[i];
                }
            }
        }
        public void OptimizeNormalizer(double value)
        {
            if (value < min[0])
            {
                min[0] = value;
            }
            if (value > max[0])
            {
                max[0] = value;
            }   
        }

        public void Normalize01(double[] tuple, bool optimize)
        {
            if(optimize) OptimizeNormalizer(tuple);
            for (int i = 0; i < tuple.Length; i++)
            {
                tuple[i] = (tuple[i] - min[i]) / (max[i] - min[i] + 1e-8);
            }
        }
        public void Normalize01(List<double> tuple, bool optimize)
        {
            if (optimize) OptimizeNormalizer(tuple.ToArray());
            for (int i = 0; i < tuple.Count; i++)
            {
                tuple[i] = (tuple[i] - min[i]) / (max[i] - min[i] + 1e-8);
            }
        }

        public void NormalizeMinusOneOne(double[] tuple, bool optimize)
        {
            if(optimize) OptimizeNormalizer(tuple);
            for (int i = 0; i < tuple.Length; i++)
            {
                tuple[i] = 2 * (tuple[i] - min[i]) / (max[i] - min[i] + 1e-8) - 1;
            }
        }
        public void NormalizeMinusOneOne(List<double> tuple, bool optimize)
        {
            if (optimize) OptimizeNormalizer(tuple.ToArray());
            for (int i = 0; i < tuple.Count; i++)
            {
                tuple[i] = 2 * (tuple[i] - min[i]) / (max[i] - min[i] + 1e-8) - 1;
            }
        }

        public void Normalize01(ref double value, bool optimize)
        {
            if (optimize) OptimizeNormalizer(new double[] { value });
            value = (value - min[0]) / (max[0] - min[0] + 1e-8);
        }
        public void NormalizeMinusOneOne(ref double value, bool optimize)
        {
            if (optimize) OptimizeNormalizer(new double[] { value });
            value = 2 * (value - min[0]) / (max[0] - min[0] + 1e-8) - 1;
        }
    }
}