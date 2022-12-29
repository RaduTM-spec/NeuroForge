using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEditor;
using UnityEngine;


namespace SmartAgents {

    [Serializable]
    public class ArtificialNeuralNetwork: ScriptableObject
    {
        [SerializeField] public int[] format;
        [SerializeField] public NeuronLayer[] neuronLayers;
        [SerializeField] public WeightLayer[] weightLayers;
        [SerializeField] public BiasLayer[] biasLayers;

        private WeightLayer[] weightGradients = null;
        private BiasLayer[] biasGradients = null;
        private WeightLayer[] weightsMomentum = null;
        private BiasLayer[] biasesMomentum = null;

        LossType lossType;


        public ArtificialNeuralNetwork(int inputs, int outputs, HiddenLayers size)
        {
            //set format
            switch(size)
            {
                case HiddenLayers.None:
                    format = new int[2];
                    format[0] = inputs;
                    format[1] = outputs;
                    break;
                case HiddenLayers.OneSmall:
                    format = new int[3];
                    format[0] = inputs;
                    format[1] = (inputs+outputs)/2;
                    format[2] = outputs;
                    break;
                case HiddenLayers.OneLarge:
                    format = new int[3];
                    format[0] = inputs;
                    format[1] = inputs + outputs;
                    format[2] = outputs;
                    break;
                case HiddenLayers.TwoSmall:
                    format = new int[4];
                    format[0] = inputs;
                    format[1] = (inputs + outputs)/2;
                    format[2] = (outputs + outputs)/2;
                    format[3] = outputs;
                    break;
                case HiddenLayers.TwoLarge:
                    format = new int[4];
                    format[0] = inputs;
                    format[1] = inputs + outputs;
                    format[2] = outputs + outputs;
                    format[3] = outputs;
                    break;
                
                default: //None
                    format = new int[2];
                    format[0] = inputs;
                    format[1] = outputs;
                    break;
            }

            //init by format
            {
                neuronLayers = new NeuronLayer[format.Length];
                biasLayers = new BiasLayer[format.Length];
                weightLayers = new WeightLayer[format.Length - 1];

                for (int i = 0; i < neuronLayers.Length; i++)
                {
                    neuronLayers[i] = new NeuronLayer(format[i]);
                    biasLayers[i] = new BiasLayer(format[i]);

                }
                for (int i = 0; i < neuronLayers.Length - 1; i++)
                {
                    weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1]);
                }
            }
        }
        public void Save()
        {
            string name = "Network#" + UnityEngine.Random.Range(1, 1000);
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/Neural_Networks/" + name + ".asset");
            EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/Icons/network_icon.png"));
        }

        //-------------------------------------------------------------------------------------------------------//
        public double[] ForwardPropagation(double[] inputs)
        {
            neuronLayers[0].SetValues(inputs);//biases are ignored
            for (int l = 1; l < neuronLayers.Length; l++)
            {
                for (int n = 0; n < neuronLayers[l].neurons.Length; n++)
                {
                    double sumValue = biasLayers[l].biases[n];
                    for (int prevn = 0; prevn < neuronLayers[l-1].neurons.Length; prevn++)
                    {
                        sumValue += neuronLayers[l - 1].neurons[prevn].value * weightLayers[l - 1].weights[prevn][n];
                    }
                    sumValue = Functions.Activation.Tanh(sumValue);
                    neuronLayers[l].neurons[n].value = sumValue;
                }
            }
            return neuronLayers[neuronLayers.Length - 1].GetValues();
        }
        public void BackPropagation(double[] inputs, double[] labels, bool applyGradients, float learningRate = 0.1f , float momentum = 0.9f, float regularization = 0.01f)
        {
            if (weightGradients== null)
                InitGradients();

            //Update Gradients

            if(applyGradients)
            {
                //Apply Gradients
            }
            void InitGradients()
            {
                biasGradients = new BiasLayer[format.Length];
                biasesMomentum = new BiasLayer[format.Length];
                weightGradients = new WeightLayer[format.Length - 1];
                weightsMomentum = new WeightLayer[format.Length - 1];

                for (int i = 0; i < neuronLayers.Length; i++)
                {
                    biasGradients[i] = new BiasLayer(format[i]);
                    biasesMomentum[i] = new BiasLayer(format[i]);

                }
                for (int i = 0; i < neuronLayers.Length - 1; i++)
                {
                    weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1]);
                    weightsMomentum[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1]);
                }

            }
        }

        //--------------------------------------------------------------------------------------------------------//

        
    }

   
}