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

        [SerializeField] public ActivationType activationType = ActivationType.Tanh;
        [SerializeField] public ActivationType outputActivationType = ActivationType.Tanh;
        [SerializeField] public LossType lossType = LossType.MeanSquare;

        private WeightLayer[] weightGradients = null;
        private WeightLayer[] weightMomentum = null;
        private BiasLayer[] biasGradients = null;    
        private BiasLayer[] biasMomentum = null;

        
        

        List<Sample> samples = new List<Sample>();

        public ArtificialNeuralNetwork(int inputs, int outputs, HiddenLayers size, ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction)
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
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/network_icon.png"));
        }

        //-------------------------------------------------------------------------------------------------------//
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
        public void BackPropagation(double[] inputs, double[] labels, bool applyGradients, float learningRate = 0.1f , float momentum = 0.9f, float regularization = 0.01f)
        {
            if (weightGradients == null)
                InitGradients();

            double[] predictions = ForwardPropagation(inputs);
            double error = Functions.Cost.CalculateOutputLayerCost(neuronLayers[neuronLayers.Length-1], labels, outputActivationType, lossType);

            for (int i = neuronLayers.Length - 2; i > 0; i--)
            {
                Functions.Cost.CalculateLayerCost(neuronLayers[i], weightLayers[i - 1], neuronLayers[i + 1], activationType);
                UpdateGradientsOf(weightLayers[i - 1], biasLayers[i], neuronLayers[i-1], neuronLayers[i]);
            }
            if (applyGradients) {; }
                //
            //implemented collected modified learnRate in order to apply the gradients
            
        }

        //--------------------------------------------------------------------------------------------------------//
        private void UpdateGradientsOf(WeightLayer weightGradient, BiasLayer biasGradient, NeuronLayer previousNeuronLayer, NeuronLayer nextNeuronLayer)
        {
            lock(weightGradient)
            {
                for (int i = 0; i < previousNeuronLayer.neurons.Length; i++)
                {
                    for (int j = 0; j < nextNeuronLayer.neurons.Length ; j++)
                    {
                        weightGradient.weights[i][j] += previousNeuronLayer.neurons[i].OutValue * nextNeuronLayer.neurons[i].CostValue;
                    }
                }
            }
            lock(biasGradient)
            {
                for (int i = 0; i < previousNeuronLayer.neurons.Length; i++)
                {
                    biasGradient.biases[i] += 1 * nextNeuronLayer.neurons[i].CostValue;
                }
            }
        }
       
        private void ApplyGradients(float modifiedLearnRate, float momentum, float regularization)
        {
            double weightDecay = 1 - regularization * modifiedLearnRate;
            for (int i = 0; i < weightLayers.Length; i++)
            {
                for (int j = 0; j < weightLayers[i].weights.Length; j++)
                {
                    for (int k = 0; k < weightLayers[i].weights[j].Length; k++)
                    {
                        weightMomentum[i].weights[j][k] = weightMomentum[i].weights[j][k] * momentum - weightGradients[i].weights[j][k];
                        weightLayers[i].weights[j][k] = weightLayers[i].weights[j][k] * weightDecay + weightMomentum[i].weights[j][k];
                        weightGradients[i].weights[j][k] = 0;
                    }
                }
            }
            for (int i = 0; i < biasLayers.Length; i++)
            {
                for (int j = 0; j < biasLayers[i].biases.Length; j++)
                {
                    biasMomentum[i].biases[j] = biasMomentum[i].biases[j] * momentum - biasGradients[i].biases[j];
                    biasLayers[i].biases[j] += biasMomentum[i].biases[j];
                    biasGradients[i].biases[j] = 0;
                }
            }
        }
        void InitGradients()
        {
            biasGradients = new BiasLayer[format.Length];
            biasMomentum = new BiasLayer[format.Length];
            weightGradients = new WeightLayer[format.Length - 1];
            weightMomentum = new WeightLayer[format.Length - 1];

            for (int i = 0; i < neuronLayers.Length; i++)
            {
                biasGradients[i] = new BiasLayer(format[i],true);
                biasMomentum[i] = new BiasLayer(format[i],true);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], true);
                weightMomentum[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], true);
            }

        }

        //--------------------------------------------------------------------------------------------------------//

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