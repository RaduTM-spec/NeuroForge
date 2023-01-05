using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEditor;
using UnityEngine;
using UnityEngine.Windows;
using static System.Collections.Specialized.BitVector32;

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
        public ArtificialNeuralNetwork(int inputs, int outputs, HiddenLayers hiddenSize, ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction, bool createAsset, string name)
        {
            //DECIDE FORMAT
            switch(hiddenSize)
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

            if (!createAsset)
                return;
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
            //EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/network_icon.png"));*/
            
        }
        public ArtificialNeuralNetwork(ArtificialNeuralNetwork other, bool createAsset, string name)
        {
            this.format = new int[other.format.Length];
            this.neuronLayers = new NeuronLayer[other.neuronLayers.Length];
            this.biasLayers = new BiasLayer[other.biasLayers.Length];
            this.weightLayers = new WeightLayer[other.weightLayers.Length];

            for (int i = 0; i < neuronLayers.Length; i++)
            {
                neuronLayers[i] = new NeuronLayer(format[i]);
                biasLayers[i] = new BiasLayer(format[i]);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1]);
            }

            SetParametersFrom(other);

            if (!createAsset)
                return;
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
        }
        public void SetParametersFrom(ArtificialNeuralNetwork other)
        {
            for (int i = 0; i < format.Length; i++)
            {
                this.format[i] = other.format[i];
            }
            for (int i = 0; i < neuronLayers.Length; i++)
            {
                this.neuronLayers[i] = (NeuronLayer)other.neuronLayers[i].Clone();
            }
            for (int i = 0; i < biasLayers.Length; i++)
            {
                this.biasLayers[i] = (BiasLayer)other.biasLayers[i].Clone();
            }
            for (int i = 0; i < weightLayers.Length; i++)
            {
                this.weightLayers[i] = (WeightLayer)other.weightLayers[i].Clone();
            }

            this.activationType = other.activationType;
            this.outputActivationType = other.outputActivationType;
            this.lossType = other.lossType;
        }

        #region TRAIN
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
        public double BackwardPropagation(double[] inputs, double[] labels)
        {
            if (weightGradients == null || weightGradients.Length == 0)
                InitGradients_InitMomentums();

            ForwardPropagation(inputs);
            double error = Functions.Cost.CalculateOutputLayerCost(neuronLayers[neuronLayers.Length-1], labels, outputActivationType, lossType);

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                Functions.Cost.CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1], activationType);
            }
            backPropagationsCount++;
            return error;
        }
        public void BackwardPropagation(double[] inputs, double surrogateLoss)
        {
            if (weightGradients == null || weightGradients.Length == 0)
                InitGradients_InitMomentums();
            double[] outputs = ForwardPropagation(inputs);
            NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];

            //Calculate the error
            for (int i = 0; i < outLayer.neurons.Length; i++)
            {
                outLayer.neurons[i].CostValue = (outputs[i] - surrogateLoss) * Functions.Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
            }

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                Functions.Cost.CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1], activationType);
            }
            backPropagationsCount++;
        }
        public void UpdateParameters(float learningRate, float momentum, float regularization)
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

        #endregion


        #region OTHER

        public static double[] GetLogProb(double[] rawOutputs)
        {
            double[] means = new double[rawOutputs.Length / 2];
            double[] stds = new double[rawOutputs.Length / 2];
            for (int i = 0; i < rawOutputs.Length; i += 2)
            {
                means[i / 2] = rawOutputs[i];
                stds[i / 2] = rawOutputs[i + 1] + 0.00000001;
            }


            double[] actions = new double[rawOutputs.Length / 2];
            for (int i = 0; i < actions.Length / 2; i++)
            {

                double mean = means[i];
                double std = stds[i];
                actions[i] = Functions.RandomGaussian(mean, std);
            }

            double[] log_probs = new double[rawOutputs.Length / 2];
            for (int i = 0; i < log_probs.Length; i++)
            {
                double mean = means[i];
                double std = stds[i];
                double act = actions[i];

                // probably mulitply the log with 0.5
                log_probs[i] = -Math.Log(2 * Math.PI * std * std) - Math.Pow(act - mean, 2) / (2 * std * std);
            }

            return log_probs;
        }
        public double[] GetGaussianLogProb(double[] inputs)
        {
            double[] rawOutputs = ForwardPropagation(inputs);
            double[] means = new double[rawOutputs.Length /2];
            double[] stds = new double[rawOutputs.Length / 2];
            for (int i = 0; i < rawOutputs.Length; i+=2)
            {
                means[i/2] = rawOutputs[i];
                stds[i/2] = rawOutputs[i+1] + 0.00000001;
            }

            
            double[] actions = new double[rawOutputs.Length / 2];
            for (int i = 0; i < actions.Length/2; i++)
            {
                
                double mean = means[i];
                double std = stds[i];
                actions[i] = Functions.RandomGaussian(mean, std);
            }

            double[] log_probs = new double[rawOutputs.Length / 2];
            for (int i = 0; i < log_probs.Length; i++)
            {
                double mean = means[i];
                double std = stds[i];
                double act = actions[i];

                // probably mulitply the log with 0.5
                log_probs[i] = -Math.Log(2 * Math.PI * std * std) - Math.Pow(act - mean, 2) / (2 * std * std);
            }

            return log_probs;
        }

        public int GetInputsNumber()
        {
            return format[0];
        }
        public int GetOutputsNumber()
        {
            return format[format.Length - 1];
        }
        #endregion
    }

    public enum Optimizer
    {
        GD,
        Adam//not implemented
    }

}