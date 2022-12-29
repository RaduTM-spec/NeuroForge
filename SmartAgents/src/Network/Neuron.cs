using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class Neuron
    {
        [SerializeField] public double InValue;
        [SerializeField] public double value;
        [SerializeField] public double OutValue;

        public Neuron()
        {
            InValue = 0;
            value = 0;
            OutValue = 0;
        }
    }
}