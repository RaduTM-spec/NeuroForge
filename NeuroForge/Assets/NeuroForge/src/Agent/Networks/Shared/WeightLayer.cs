using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class WeightLayer: ISerializationCallbackReceiver, ICloneable
    {
        public double[][] weights;

        //Only for serialization
        [SerializeField] private List<double> serializedWeights;
        [SerializeField] private int prevNeurons;
        [SerializeField] private int nextNeurons;

        public WeightLayer(NeuronLayer firstLayer, NeuronLayer secondLayer, InitializationType initType)
        {
            weights = new double[firstLayer.neurons.Length][];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = new double[secondLayer.neurons.Length];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    switch(initType)
                    {
                        case InitializationType.Zero:
                            weights[i][j] = 0;
                            break;
                        case InitializationType.NormalDistribution:
                            weights[i][j] = Functions.RandomGaussian();
                            break;
                        case InitializationType.Xavier:
                            weights[i][j] = Functions.RandomGaussian(0, Math.Sqrt(2.0 /(firstLayer.neurons.Length + secondLayer.neurons.Length)));
                            break;
                        case InitializationType.He:
                            weights[i][j] = Functions.RandomGaussian(0, Math.Sqrt(2.0 / firstLayer.neurons.Length));
                            break;
                    }
                }
            }

        }
        private WeightLayer() { }
        public void Zero()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = 0;
                }
            }
        }
        public object Clone()
        {
            WeightLayer clone = new WeightLayer();
            clone.weights = new double[this.weights.Length][];
            for (int i = 0; i < this.weights.Length; i++)
            {
                clone.weights[i] = new double[this.weights[i].Length];
                for (int j = 0; j < this.weights[i].Length; j++)
                {
                    clone.weights[i][j] = this.weights[i][j];
                }
            }

            clone.prevNeurons = this.prevNeurons;
            clone.nextNeurons = this.nextNeurons;

            return clone;
        }

        public void OnBeforeSerialize()
        {
            serializedWeights = new List<double>();
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    serializedWeights.Add(weights[i][j]);
                }
            }
            prevNeurons = weights.Length;
            nextNeurons = weights[0].Length;
        }
        public void OnAfterDeserialize()
        {
            int index = 0;
            weights = new double[prevNeurons][];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = new double[nextNeurons];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = serializedWeights[index++];
                }
            }

            serializedWeights.Clear();
        }

    }
}