using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class BiasLayer
    {
        [SerializeField] public double[] biases;
        public BiasLayer(int noBiases)
        {
            biases = new double[noBiases];
        }
    }
}
