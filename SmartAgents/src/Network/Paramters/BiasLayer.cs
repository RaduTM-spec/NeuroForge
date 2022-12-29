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
        public BiasLayer(int noBiases, bool zeroes = false)
        {
            biases = new double[noBiases];
            for (int i = 0; i < biases.Length; i++)
            {
                if(zeroes)
                    biases[i] = 0;
                else
                    biases[i] = Functions.RandomGaussian();
            }
        }
    }
}
