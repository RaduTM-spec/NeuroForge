using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.Design;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class BiasLayer : ICloneable
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
        public object Clone()
        {
            BiasLayer clone = new BiasLayer(biases.Length);
            clone.biases = new double[this.biases.Length];
            for (int i = 0; i < clone.biases.Length; i++)
            {
                clone.biases[i] = this.biases[i];
            }
            return clone;
        }
    }
}
