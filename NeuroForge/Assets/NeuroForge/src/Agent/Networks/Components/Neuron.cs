using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class Neuron : ICloneable
    {
        [SerializeField] public double InValue;
        [SerializeField] public double CostValue;
        [SerializeField] public double OutValue;

        public Neuron()
        {
            InValue = 0;
            CostValue = 0;
            OutValue = 0;
        }

        public object Clone()
        {
            return new Neuron
            {
                InValue = this.InValue,
                CostValue = this.CostValue,
                OutValue = this.OutValue
            };
        }

    }
}