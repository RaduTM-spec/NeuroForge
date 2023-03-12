using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static UnityEditor.Experimental.GraphView.GraphView;
namespace NeuroForge
{
    [Serializable]
    public class NeuronLayer : ICloneable, IResetable
    {
        [SerializeField] public Neuron[] neurons;
        [SerializeField] public ActivationType activationType;
        public NeuronLayer(int noNeurons, ActivationType activationType)
        {
            this.activationType = activationType;
            neurons = new Neuron[noNeurons];
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron();
            }
        }
        public object Clone()
        {
            NeuronLayer clone = new NeuronLayer(this.neurons.Length, this.activationType);
            for (int i = 0; i < this.neurons.Length; i++)
            {
                clone.neurons[i] = (Neuron)this.neurons[i].Clone();
            }
            return clone;
        }
        public void Activate()
        {
            if (activationType == ActivationType.SoftMax)
            {
                double[] InValuesToActivate = neurons.Select(x => x.InValue).ToArray();
                Functions.Activation.SoftMax(InValuesToActivate);
                for (int i = 0; i < InValuesToActivate.Length; i++)
                {
                    neurons[i].OutValue = InValuesToActivate[i];
                }
            }
            else
            {
                foreach (Neuron neuron in neurons)
                {
                    neuron.OutValue = Functions.Activation.ActivateValue(neuron.InValue, activationType);
                }
            }
        }

        public void SetInValues(double[] values)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].InValue = values[i];
            }
        }
        public void SetCostValues(double[] values)
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
        public double[] GetCostValues()
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

        public void Reset()
        {
            foreach (var neur in neurons)
            {
                neur.InValue = 0;
                neur.OutValue = 0;
                neur.CostValue = 0;
            }
        }
    }
}