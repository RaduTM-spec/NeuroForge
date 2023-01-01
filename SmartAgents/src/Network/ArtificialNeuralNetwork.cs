using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
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

        [SerializeField] public ActivationType activationType = ActivationType.Tanh;
        [SerializeField] public ActivationType outputActivationType = ActivationType.Tanh;
        [SerializeField] public LossType lossType = LossType.MeanSquare;

        private WeightLayer[] weightGradients;
        private WeightLayer[] weightMomentums;
        private BiasLayer[] biasGradients;    
        private BiasLayer[] biasMomentums;

        int backPropagationsCount = 0;
        public ArtificialNeuralNetwork(int inputs, int outputs, HiddenLayers size, ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction, string name)
        {
            //DECIDE FORMAT
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
                    format[2] = (inputs + outputs)/2;
                    format[3] = outputs;
                    break;
                case HiddenLayers.TwoLarge:
                    format = new int[4];
                    format[0] = inputs;
                    format[1] = inputs + outputs;
                    format[2] = inputs + outputs;
                    format[3] = outputs;
                    break;
                
                default: //None
                    format = new int[2];
                    format[0] = inputs;
                    format[1] = outputs;
                    break;
            }

            //CONSTRUCTOR
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


            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
            EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/network_icon.png"));
            
        }
       

        public double[] ForwardPropagation(double[] inputs)
        {
            neuronLayers[0].SetOutValues(inputs);
            for (int l = 1; l < neuronLayers.Length; l++)
            {
                for (int n = 0; n < neuronLayers[l].neurons.Length; n++)
                {
                    double sumValue = biasLayers[l].biases[n];
                    for (int prevn = 0; prevn < neuronLayers[l-1].neurons.Length; prevn++)
                    {
                        sumValue += neuronLayers[l - 1].neurons[prevn].OutValue * weightLayers[l - 1].weights[prevn][n];
                    }
                    neuronLayers[l].neurons[n].InValue = sumValue;
                }

                //Activate neuron layer
                if(l < neuronLayers.Length - 1)
                {
                    Functions.Activation.ActivateLayer(neuronLayers[l], activationType);
                }
                //Activate output neuron layer
                else
                {
                    Functions.Activation.ActivateLayer(neuronLayers[l], outputActivationType);
                }
            }


            return neuronLayers[neuronLayers.Length - 1].GetOutValues();
        }
        public double BackPropagation(double[] inputs, double[] labels, double advantageEstimate = 1)
        {
            if (weightGradients == null || weightGradients.Length == 0)
                InitGradients_InitMomentums();

            ForwardPropagation(inputs);
            double error = Functions.Cost.CalculateOutputLayerCost(neuronLayers[neuronLayers.Length-1], labels, outputActivationType, lossType, advantageEstimate);

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                Functions.Cost.CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1], activationType);
            }
            backPropagationsCount++;
            return error;
        }
        public void UpdateParameters(float learningRate = 0.1f, float momentum = 0.9f, float regularization = 0.001f)
        {
            ApplyGradients(learningRate / backPropagationsCount, momentum, regularization);
            backPropagationsCount = 0;
        }


        private void InitGradients_InitMomentums()
        {
            biasGradients = new BiasLayer[format.Length];
            biasMomentums = new BiasLayer[format.Length];
            weightGradients = new WeightLayer[format.Length - 1];
            weightMomentums = new WeightLayer[format.Length - 1];

            for (int i = 0; i < neuronLayers.Length; i++)
            {
                biasGradients[i] = new BiasLayer(format[i], true);
                biasMomentums[i] = new BiasLayer(format[i], true);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], true);
                weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], true);
            }


        }
        private void UpdateGradients(WeightLayer weightGradient, BiasLayer biasGradient, NeuronLayer previousNeuronLayer, NeuronLayer nextNeuronLayer)
        {
            //Related to Backpropagation
            lock(weightGradient)
            {
                for (int i = 0; i < previousNeuronLayer.neurons.Length; i++)
                {
                   
                    for (int j = 0; j < nextNeuronLayer.neurons.Length ; j++)
                    {
                        weightGradient.weights[i][j] += previousNeuronLayer.neurons[i].OutValue * nextNeuronLayer.neurons[j].CostValue;
                    }
                }
            }
            lock(biasGradient)
            {
                for (int i = 0; i < nextNeuronLayer.neurons.Length; i++)
                {
                    biasGradient.biases[i] += 1 * nextNeuronLayer.neurons[i].CostValue;
                }
            }
        }
        private void ApplyGradients(float modifiedLearnRate, float momentum, float regularization)
        {
            //Related to UpdateParameters
            double weightDecay = 1 - regularization * modifiedLearnRate;
            for (int l = 0; l < weightLayers.Length; l++)
            {
                for (int i = 0; i < weightLayers[l].weights.Length; i++)
                {
                    for (int j = 0; j < weightLayers[l].weights[i].Length; j++)
                    {
                        weightMomentums[l].weights[i][j] = weightMomentums[l].weights[i][j] * momentum - weightGradients[l].weights[i][j] * modifiedLearnRate;
                        weightLayers[l].weights[i][j] = weightLayers[l].weights[i][j] * weightDecay + weightMomentums[l].weights[i][j];
                        weightGradients[l].weights[i][j] = 0;
                    }
                }
            }
            for (int i = 0; i < biasLayers.Length; i++)
            {
                for (int j = 0; j < biasLayers[i].biases.Length; j++)
                {
                    biasMomentums[i].biases[j] = biasMomentums[i].biases[j] * momentum - biasGradients[i].biases[j] * modifiedLearnRate;
                    biasLayers[i].biases[j] += biasMomentums[i].biases[j];
                    biasGradients[i].biases[j] = 0;
                }
            }
        }
        

        public int GetInputsNumber()
        {
            return format[0];
        }
        public int GetOutputsNumber()
        {
            return format[format.Length - 1];
        }       
    }

   
}