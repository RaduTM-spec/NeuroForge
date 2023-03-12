using NeuroForge;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class SequencialNeuronLayer : NeuronLayer, IResetable
    {
        [SerializeField] double[] sequencialWeights;
        [SerializeField] InitializationType initializationType;
        public SequencialNeuronLayer(int noNeurons, ActivationType activType, InitializationType initType) : base(noNeurons, activType)
        {
            // Initialize seqWeights
            sequencialWeights = new double[noNeurons];

            // Take each initType.. but for now i will init them like this
            for (int i = 0; i < sequencialWeights.Length; i++)
            {
                sequencialWeights[i] = Functions.RandomGaussian(1, 0.2f);
            }

        }

        public new object Clone()
        {
            SequencialNeuronLayer clone = new SequencialNeuronLayer(this.neurons.Length, this.activationType, this.initializationType);
            for (int i = 0; i < this.neurons.Length; i++)
            {
                clone.neurons[i] = this.neurons[i].Clone() as Neuron;
                clone.sequencialWeights[i] = this.sequencialWeights[i];
            }
            return clone;
        }
        public new void Activate()
        {
            // Spark the neuron
            foreach (var neur in neurons)
            {
                neur.OutValue = Functions.Activation.ActivateValue(neur.InValue, activationType);
            }

            // Retrieve the output to input again
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].InValue += neurons[i].OutValue * sequencialWeights[i];
            }

        }

       
    }

}
