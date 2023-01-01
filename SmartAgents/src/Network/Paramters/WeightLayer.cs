using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class WeightLayer: ISerializationCallbackReceiver
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