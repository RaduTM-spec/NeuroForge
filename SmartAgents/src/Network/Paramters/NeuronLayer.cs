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

        public void SetInValues(double[] values)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].InValue = values[i];
            }
        }
        public void SetValues(double[] values)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].CostValue = values[i];
            }
        }
        public void SetOutValues(double[] values)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].OutValue = values[i];
            }
        }

        public double[] GetInValues()
        {
            double[] inVals = new double[neurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                inVals[i] = neurons[i].InValue;
            }
            return inVals;
        }
        public double[] GetValues()
        {
            double[] vals = new double[neurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                vals[i] = neurons[i].CostValue;
            }
            return vals;
        }
        public double[] GetOutValues()
        {
            double[] vals = new double[neurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                vals[i] = neurons[i].OutValue;
            }
            return vals;
        }
    }
}