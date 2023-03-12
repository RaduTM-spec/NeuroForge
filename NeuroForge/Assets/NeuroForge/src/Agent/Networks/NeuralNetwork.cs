using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Xsl;
using Unity.VisualScripting;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEditor;
using UnityEngine;
using UnityEngine.Windows;
using static NeuroForge.Functions;
using static System.Collections.Specialized.BitVector32;

namespace NeuroForge {

    [Serializable]
    public class NeuralNetwork: ScriptableObject
    {
        [SerializeField] public int[] layerFormat;
        [SerializeField] public NeuronLayer[] neuronLayers;
        [SerializeField] public WeightLayer[] weightLayers;
        [SerializeField] public BiasLayer[] biasLayers;

        [SerializeField] public InitializationType initializationType; // shows how the network was initialized
        [SerializeField] public ActivationType activationType;
        [SerializeField] public ActivationType outputActivationType;
        [SerializeField] public LossType lossType;

        private WeightLayer[] weightGradients;
        private WeightLayer[] weightMomentums;
        private BiasLayer[] biasGradients;    
        private BiasLayer[] biasMomentums;

        int backpropsCount = 0;
        public NeuralNetwork(int inputs, int outputs,int hiddenUnits, int hiddenLayersNumber, 
                             ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction, 
                             InitializationType initType, bool createAsset, string name) 
        {
            this.layerFormat = GetFormat(inputs, outputs, hiddenUnits, hiddenLayersNumber);
            this.initializationType = initType;
            this.activationType = activationFunction;
            this.outputActivationType = outputActivationFunction;
            this.lossType = lossFunction;

            neuronLayers = new NeuronLayer[layerFormat.Length];
            biasLayers = new BiasLayer[layerFormat.Length];
            weightLayers = new WeightLayer[layerFormat.Length - 1];

            for (int i = 0; i < neuronLayers.Length; i++)
            {
                if (i != neuronLayers.Length - 1)
                    neuronLayers[i] = new NeuronLayer(layerFormat[i], activationType);
                else
                    neuronLayers[i] = new NeuronLayer(layerFormat[i], outputActivationType);
                biasLayers[i] = new BiasLayer(layerFormat[i], initType);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType); 
            }

            if (createAsset)
            {
                Debug.Log(name + " was created!");
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
                AssetDatabase.SaveAssets();
            }
        }

        
        public double[] Forward(double[] inputs)
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

                neuronLayers[l].Activate();
            }
            return neuronLayers[neuronLayers.Length - 1].GetOutValues();
        }
        public double Backward(double[] inputs, double[] labels)
        {
            if (weightGradients == null || weightGradients.Length < 1)
                ZeroGrad();

            Forward(inputs);
            double error = CalculateOutputLayerCost(labels);

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1]);
            }
            backpropsCount++;
            return error;
        }
        public void GradClipNorm(float threshold)
        {
            double global_sum = 0;

            // Sum weights' gradients
            foreach (var grad_layer in weightGradients)
            {
                foreach (var clump in grad_layer.weights)
                {
                    foreach (var w_grad in clump)
                    {
                        global_sum += w_grad * w_grad;
                    }
                }
            }

            // Sum biases' gradients
            foreach (var bias_layer in biasGradients)
            {
                foreach (var b_grad in bias_layer.biases)
                {
                    global_sum += b_grad * b_grad;
                }
            }

            double scalar = threshold / Math.Max(threshold, global_sum);

            // Normalize weights
            for (int lay = 0; lay < weightGradients.Length; lay++)
            {
                for (int i = 0; i < weightGradients[lay].weights.Length; i++)
                {
                    for (int j = 0; j < weightGradients[lay].weights[i].Length; j++)
                    {
                        weightGradients[lay].weights[i][j] *= scalar;
                    }
                }
            }

            // Normalize biases
            for (int lay = 0; lay < biasGradients.Length; lay++)
            {
                for (int i = 0; i < biasGradients[lay].biases.Length; i++)
                {
                    biasGradients[lay].biases[i] *= scalar;
                }
            }
        }
        public void OptimStep(float learningRate, float momentum, float regularization)
        {
            learningRate /= backpropsCount;
            backpropsCount = 0;

            double weightDecay = 1 - regularization * learningRate;
            for (int l = 0; l < weightLayers.Length; l++)
            {
                for (int i = 0; i < weightLayers[l].weights.Length; i++)
                {
                    for (int j = 0; j < weightLayers[l].weights[i].Length; j++)
                    {
                        double weight = weightLayers[l].weights[i][j];
                        double veloc = weightMomentums[l].weights[i][j] * momentum - weightGradients[l].weights[i][j] * learningRate;

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
                    double veloc = biasMomentums[i].biases[j] * momentum - biasGradients[i].biases[j] * learningRate;

                    biasMomentums[i].biases[j] = veloc;
                    biasLayers[i].biases[j] += veloc;

                    biasGradients[i].biases[j] = 0;
                }
            }
        }
        private void ZeroGrad()
        {
                biasGradients = new BiasLayer[layerFormat.Length];
                biasMomentums = new BiasLayer[layerFormat.Length];
                weightGradients = new WeightLayer[layerFormat.Length - 1];
                weightMomentums = new WeightLayer[layerFormat.Length - 1];

                for (int i = 0; i < neuronLayers.Length; i++)
                {
                    biasGradients[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);
                    biasMomentums[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);

                }
                for (int i = 0; i < weightLayers.Length; i++)
                {
                    weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                    weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                }          
        }

        private void UpdateGradients(WeightLayer weightGradient, BiasLayer biasGradient, NeuronLayer previousNeuronLayer, NeuronLayer nextNeuronLayer)
        {
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
        private void CalculateLayerCost(NeuronLayer layer, WeightLayer weights, NeuronLayer nextLayer)
        {
            for (int i = 0; i < layer.neurons.Length; i++)
            {
                double costVal = 0;
                for (int j = 0; j < nextLayer.neurons.Length; j++)
                {
                    costVal += nextLayer.neurons[j].CostValue * weights.weights[i][j];
                }
                costVal *= Derivative.DeriveValue(layer.neurons[i].InValue, activationType);

                layer.neurons[i].CostValue = costVal;
            }
        }
        private double CalculateOutputLayerCost(double[] labels)
        {
            NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];
            double err = 0;
            if (outputActivationType != ActivationType.SoftMax)
            {
                for (int i = 0; i < outLayer.neurons.Length; i++)
                {
                    switch (lossType)
                    {
                        case LossType.MeanSquare:
                            outLayer.neurons[i].CostValue = Loss.MeanSquareDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                            err += Error.MeanSquare(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.CrossEntropy:
                            outLayer.neurons[i].CostValue = Loss.CrossEntropyDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                            err += Error.CrossEntropy(outLayer.neurons[i].OutValue, labels[i]);                          
                            break;
                        case LossType.MeanAbsolute:
                            outLayer.neurons[i].CostValue = Loss.AbsoluteDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                            err += Error.MeanAbsolute(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                    }
                }
            }
            else
            {
                double[] derivedInValuesBySoftMax = new double[labels.Length];
                for (int i = 0; i < derivedInValuesBySoftMax.Length; i++)
                    derivedInValuesBySoftMax[i] = outLayer.neurons[i].InValue;

                Derivative.SoftMax(derivedInValuesBySoftMax);

                for (int i = 0; i < outLayer.neurons.Length; i++)
                {
                    switch (lossType)
                    {
                        case LossType.MeanSquare:
                            outLayer.neurons[i].CostValue = Loss.MeanSquareDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                            err += Error.MeanSquare(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.CrossEntropy:
                            outLayer.neurons[i].CostValue = Loss.CrossEntropyDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                            err += Error.CrossEntropy(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.MeanAbsolute:
                            outLayer.neurons[i].CostValue = Loss.AbsoluteDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                            err += Error.MeanAbsolute(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                    }
                }



            }

            return err / labels.Length;
        }
       

        // OTHER
        static int[] GetFormat(int inputs, int outs, int hidden_units, int hidden_lay_num)
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
        public int GetNoInputs() => layerFormat[0];
        public int GetNoOutputs() => layerFormat[layerFormat.Length - 1];
    }


}