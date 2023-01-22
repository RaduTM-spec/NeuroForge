using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.Design;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class BiasLayer : ICloneable
    {
        [SerializeField] public double[] biases;
        public BiasLayer(int noBiases, InitializationType initType)
        {
            biases = new double[noBiases];
            for (int i = 0; i < biases.Length; i++)
            {
                switch (initType)
                {
                    case InitializationType.Zero | InitializationType.Xavier:
                        biases[i] = 0;
                        break;
                    case InitializationType.NormalDistribution:
                        biases[i] = Functions.RandomGaussian();
                        break;
                    case InitializationType.He:
                        biases[i] = Functions.RandomGaussian(0, 0.01);
                        break;
                }
            }
        }
        public void Zero()
        {
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] = 0;
            }
        }
        public object Clone()
        {
            BiasLayer clone = new BiasLayer(biases.Length, InitializationType.Zero);
            clone.biases = new double[this.biases.Length];
            for (int i = 0; i < clone.biases.Length; i++)
            {
                clone.biases[i] = this.biases[i];
            }
            return clone;
        }
    }
}
