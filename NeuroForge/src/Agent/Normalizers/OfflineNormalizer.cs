using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NeuroForge
{
    public class OfflineNormalizer
    {
        public static void Normalize01(List<double> list, Func<double, double> func = null)
        {
            double min = list.Min(func);
            double max = list.Max(func);
            for (int i = 0; i < list.Count; i++)
            {
                list[i] = (list[i] - min) / (max - min);
            }

        }
        
    }
}

