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
    public class NeuralNetwork: ScriptableObject
    {
        [SerializeField] public int[] format;
        [SerializeField] public int[] outputShape;
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
        public NeuralNetwork(int inputs, int outputs,int hiddenUnits, int hiddenLayersNumber, ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction, bool createAsset, string name)
        {
            this.format = GetFormat(inputs, outputs, hiddenUnits, hiddenLayersNumber);

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
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1); 
            }

            if (createAsset)
            {
                Debug.Log(name + " was created!");
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
                AssetDatabase.SaveAssets();
            }
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
                    Functions.Activation.ActivateOutputLayer(neuronLayers[l], outputActivationType, outputShape);
                }
            }


            return neuronLayers[neuronLayers.Length - 1].GetOutValues();
        }
        public double BackPropagation(double[] inputs, double[] labels)
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
        public void OptimizeParameters(float learningRate, float momentum, float regularization, bool descent)
        {
            if(descent == true)
                ApplyGradients(learningRate / backPropagationsCount, momentum, regularization, -1);
            else //ascent
                ApplyGradients(learningRate / backPropagationsCount, momentum, regularization, 1);
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
                weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1, true);
                weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1, true);
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
        private void ApplyGradients(float modifiedLearnRate, float momentum, float regularization, double direction)
        {
            //Related to UpdateParameters
            double weightDecay = 1 - regularization * modifiedLearnRate;
            for (int l = 0; l < weightLayers.Length; l++)
            {
                for (int i = 0; i < weightLayers[l].weights.Length; i++)
                {
                    for (int j = 0; j < weightLayers[l].weights[i].Length; j++)
                    {
                        double weight = weightLayers[l].weights[i][j];
                        double veloc = weightMomentums[l].weights[i][j] * momentum + weightGradients[l].weights[i][j] * modifiedLearnRate * direction;

                        weightMomentums[l].weights[i][j] = veloc;
                        weightLayers[l].weights[i][j] = weight * weightDecay + veloc;

                        //Reset the gradient
                        weightGradients[l].weights[i][j] = 0;
                    }
                }
            }
            for (int i = 0; i < biasLayers.Length; i++)
            {
                for (int j = 0; j < biasLayers[i].biases.Length; j++)
                {
                    double bias = biasLayers[i].biases[j];
                    double veloc = bias * momentum + bias * modifiedLearnRate * direction;

                    biasMomentums[i].biases[j] = veloc;
                    biasLayers[i].biases[j] += veloc;

                    biasGradients[i].biases[j] = 0;
                }
            }
        }

        #endregion


        #region OTHER
        private int[] GetFormat(int inputs, int outs, int hidden_units, int hidden_lay_num)
        {
            int[] form = new int[2 + hidden_lay_num];

            form[0] = inputs;
            for (int i = 1; i <= hidden_lay_num; i++)
            {
                form[i] = hidden_units;
            }
            form[form.Length - 1] = outs;

            return form;
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


}