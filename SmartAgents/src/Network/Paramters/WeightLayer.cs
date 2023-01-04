using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class WeightLayer: ISerializationCallbackReceiver, ICloneable
    {
        public double[][] weights;

        //Only for serialization
        [SerializeField] private List<double> serializedWeights;
        [SerializeField] private int prevNeurons;
        [SerializeField] private int nextNeurons;

        public WeightLayer(NeuronLayer firstLayer, NeuronLayer secondLayer, bool zeroes = false)
        {
            weights = new double[firstLayer.neurons.Length][];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = new double[secondLayer.neurons.Length];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    if (zeroes)
                        weights[i][j] = 0;
                    else
                        weights[i][j] = Functions.RandomGaussian();
                }
            }

        }
        private WeightLayer() { }
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