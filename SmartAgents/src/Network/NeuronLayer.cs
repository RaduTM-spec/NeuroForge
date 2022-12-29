using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class NeuronLayer
    {
        public Neuron[] neurons;

        public NeuronLayer(int noNeurons)
        {
            neurons = new Neuron[noNeurons];
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron();
            }
        }
        public void SetValues(double[] values)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].value = values[i];
            }
        }
        public double[] GetValues()
        {
            double[] values = new double[neurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                values[i] = neurons[i].value;
            }
            return values;
        }
    }
}