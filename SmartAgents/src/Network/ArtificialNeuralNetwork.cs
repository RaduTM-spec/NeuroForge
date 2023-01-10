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
        [SerializeField] public int[] outputBranchFormat;
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
        public ArtificialNeuralNetwork(int inputs, int[] outputBranchFormat, int hiddenUnits, int hiddenLayersNumber, ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction, bool createAsset, string name)
        {
            this.format = GetFormat(inputs, outputBranchFormat.Sum(), hiddenUnits, hiddenLayersNumber);
            this.outputBranchFormat = outputBranchFormat;
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
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1); //Xavier initialization is not ok
            }

            if (!createAsset)
                return;
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
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
                    Functions.Activation.ActivateOutputLayer(neuronLayers[l], outputActivationType, outputBranchFormat);
                }
            }


            return neuronLayers[neuronLayers.Length - 1].GetOutValues();
        }
        public double[] ForwardPropagation_Parallel(double[] inputs)
        {
            double[][] neurons = new double[format.Length][];
            for (int i = 0; i < format.Length; i++)
            {
                neurons[i] = new double[format[i]];
            }

            for (int lay = 1; lay < neurons.Length; lay++)
            {
                for (int neur = 0; neur < neurons[lay].Length; neur++)
                {
                    double sum = biasLayers[lay].biases[neur];
                    for (int prev = 0; prev < neurons[lay-1].Length; prev++)
                    {
                        sum += neurons[lay - 1][prev] * weightLayers[lay - 1].weights[prev][neur];
                    }
                    neurons[lay][neur] = sum;
                }
                if (lay < neurons.Length)
                    ActivateLayer(neurons[lay]);
                else
                    ActivateOutputLayer(neurons[lay]);
            }

            return neurons[neurons.Length - 1];

            void ActivateLayer(double[] layerX)
            {
                for (int i = 0; i < layerX.Length; i++)
                {
                    layerX[i] = Functions.Activation.ActivateValue(layerX[i], activationType);
                }
            }
            void ActivateOutputLayer(double[] layerX)
            {
                if (outputActivationType == ActivationType.BranchedSoftMaxActivation)
                {
                    // Get all raw values
                    List<double> rawValues = layerX.Select(x => x).ToList();

                    int index = 0;

                    // Foreach branch, activate the branch values
                    foreach (var branch in outputBranchFormat)
                    {
                        // Get the branch from raw values
                        double[] branchValues = rawValues.GetRange(index, index + branch).ToArray();

                        // Activate the branch
                        Functions.Activation.SoftMax(branchValues);

                        // Place the activated branch on OutValues
                        for (int i = index; i < index + branch; i++)
                        {
                            layerX[i] = branchValues[i - index];
                        }

                        index += branch;
                    }


                }
                else if (outputActivationType == ActivationType.PairedTanhSoftPlusActivation)
                {
                    for (int i = 0; i < layerX.Length; i++)
                    {
                        layerX[i] =
                            i % 2 == 0 ?
                            Functions.Activation.ActivateValue(layerX[i], ActivationType.Tanh) :     // mu
                            Functions.Activation.ActivateValue(layerX[i], ActivationType.SoftPlus);  // sigma
                    }
                }
                else
                {
                    for (int i = 0; i < layerX.Length; i++)
                    {
                        layerX[i] = Functions.Activation.ActivateValue(layerX[i], activationType);
                    }
                }
            }
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
        public void BackPropagation_LossCalculated(double[] inputs, double[] loss)
        {
            if (weightGradients == null || weightGradients.Length == 0)
                InitGradients_InitMomentums();
            double[] outputs = ForwardPropagation(inputs);
            NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];

            //Calculate the error
            for (int i = 0; i < outLayer.neurons.Length; i++)
            {
                outLayer.neurons[i].CostValue = (outputs[i] - loss[i]) * Functions.Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
            }

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                Functions.Cost.CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1], activationType);
            }
            backPropagationsCount++;
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
                    double veloc = biasMomentums[i].biases[j] * momentum + biasGradients[i].biases[j] * modifiedLearnRate * direction;

                    biasMomentums[i].biases[j] = veloc;
                    biasLayers[i].biases[j] += veloc;

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

        private int[] GetFormat(int inputs, int outputs, int hidden_units, int hidden_lay_num)
        {
            int[] form = new int[2 + hidden_lay_num];

            form[0] = inputs;
            for (int i = 1; i <= hidden_lay_num; i++)
            {
                form[i] = hidden_units;
            }
            form[form.Length - 1] = outputs;

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