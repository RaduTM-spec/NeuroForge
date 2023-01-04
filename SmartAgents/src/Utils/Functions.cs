using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    public struct Functions
    {
        public static double RandomGaussian(double mean = 0, double standardDeviation = 1)
        {         
            System.Random rng = new System.Random();
            float x1 = (float)(1 - rng.NextDouble());
            float x2 = (float)(1 - rng.NextDouble());

            double y1 = Mathf.Sqrt(-2.0f * Mathf.Log(x1)) * Mathf.Cos(2.0f * (float)Math.PI * x2);
            return y1 * standardDeviation + mean;
            
        }
        public struct Activation
        {
            /// <summary>
            /// The activated value of neuron.InValue is stored in neuron.OutValue
            /// </summary>
            /// <param name="neuronLayer"></param>
            /// <param name="activationFunction"></param>
            public static void ActivateLayer(NeuronLayer neuronLayer, ActivationType activationFunction)
            {
                if (activationFunction != ActivationType.SoftMax) 
                {
                    foreach (Neuron neuron in neuronLayer.neurons)
                    {
                        neuron.OutValue = ActivateValue(neuron.InValue, activationFunction);
                    }
                } 
                else
                {
                    neuronLayer.SetOutValues(SoftMax(neuronLayer.GetInValues()));
                }
            }

            private static double ActivateValue(double value, ActivationType activationFunction)
            {
                switch(activationFunction)
                {
                    case ActivationType.Tanh:
                        return Tanh(value);
                    case ActivationType.BinaryStep:
                        return BinaryStep(value);
                    case ActivationType.Sigmoid:
                        return Sigmoid(value);
                    case ActivationType.Relu:
                        return ReLU(value);
                    case ActivationType.LeakyRelu:
                        return LeakyReLU(value);
                    case ActivationType.Silu:
                        return SiLU(value);
                    default:
                        return value;
                        
                }
                
            }
            private static double BinaryStep(double value)
            {
                if (value < 0)
                    return 0;
                else return 1;
            }
            internal static double Sigmoid(double value)
            {
                return 1 / (1 + Mathf.Exp((float)-value));
            }
            private static double Tanh(double value)
            {
                return System.Math.Tanh(value);
            }
            private static double ReLU(double value)
            {
                return Mathf.Max(0, (float)value);
            }
            private static double LeakyReLU(double value, double alpha = 0.2)
            {
                if (value > 0)
                    return value;
                else return value * alpha;
            }
            private static double SiLU(double value)
            {
                return value * Sigmoid(value);
            }
            internal static double[] SoftMax(double[] values)
            {
                double sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Mathf.Exp((float)values[i]);
                    sum += values[i];
                }
                return values;
            }
        }
        public struct Derivative
        {
            /// <summary>
            /// The derivated valueof neuron.OutValue is stored in neuron.CostValue
            /// </summary>
            /// <param name="neuronLayer"></param>
            /// <param name="activationFunction"></param>
            public static void DeriveLayer(NeuronLayer neuronLayer, ActivationType activationFunction)
            {
                if (activationFunction != ActivationType.SoftMax)
                {
                    foreach (Neuron neuron in neuronLayer.neurons)
                    {
                        neuron.CostValue = DeriveValue(neuron.OutValue, activationFunction);
                    }
                }
                else
                {
                    neuronLayer.SetValues(DerivativeSoftMax(neuronLayer.GetOutValues()));
                }
            }
            static public double DeriveValue(double value, ActivationType activationFunction)
            {
                switch (activationFunction)
                {
                    case ActivationType.Tanh:
                        return DerivativeTanh(value);
                    case ActivationType.BinaryStep:
                        return DerivativeBinaryStep(value);
                    case ActivationType.Sigmoid:
                        return DerivativeSigmoid(value);
                    case ActivationType.Relu:
                        return DerivativeReLU(value);
                    case ActivationType.LeakyRelu:
                        return DerivativeLeakyReLU(value);
                    case ActivationType.Silu:
                        return DerivativeSiLU(value);
                    default:
                        return value;
                }
            }

            static public double DerivativeTanh(double  value)
            {
                return 1f - Math.Pow(Math.Tanh(value), 2);
            }
            static public double DerivativeSigmoid(double value)
            {
                return Activation.Sigmoid(value) * (1 - Activation.Sigmoid(value));
            }
            static public double DerivativeBinaryStep(double value)
            {
                return 0;
            }
            static public double DerivativeReLU(double value)
            {
                if (value < 0)
                    return 0;
                else return 1;
            }
            static public double DerivativeLeakyReLU(double value, double alpha = 0.2f)
            {
                if (value < 0)
                    return alpha;
                else return 1;
            }
            static public double DerivativeSiLU(double value)
            {
                return (1 + Mathf.Exp((float)-value) + value * Mathf.Exp((float)-value)) / Mathf.Pow((1 + Mathf.Exp((float)-value)), 2);
                //return ActivationFunctionSigmoid(value) * (1 + value * (1 - ActivationFunctionSigmoid(value))); -> works the same
            }
            static public double[] DerivativeSoftMax(double[] values)
            {
                 double[] valuesIn = new double[values.Length];
                Array.Copy(values, valuesIn, values.Length);

                Functions.Activation.SoftMax(valuesIn);//values in are now softmaxed

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = valuesIn[i] * (1 - valuesIn[i]);
                }
                return values;
            }
        }
        public struct Cost
        {
            public static double CalculateOutputLayerCost(NeuronLayer outNeurLayer, double[] expectedOuts, ActivationType outputActivation, LossType loss)
            {
                double cost = 0;
                if(outputActivation != ActivationType.SoftMax)
                {
                    for (int i = 0; i < outNeurLayer.neurons.Length; i++)
                    {
                        switch(loss)
                        {
                            case LossType.MeanSquare:
                                outNeurLayer.neurons[i].CostValue = MeanSquareDerivative(outNeurLayer.neurons[i].OutValue, expectedOuts[i]) * Derivative.DeriveValue(outNeurLayer.neurons[i].InValue, outputActivation);
                                cost += MeanSquare(outNeurLayer.neurons[i].OutValue, expectedOuts[i]);
                                break;
                            case LossType.CrossEntropy:
                                outNeurLayer.neurons[i].CostValue = CrossEntropyDerivative(outNeurLayer.neurons[i].OutValue, expectedOuts[i]) * Derivative.DeriveValue(outNeurLayer.neurons[i].InValue, outputActivation);
                                double locCost = CrossEntropy(outNeurLayer.neurons[i].OutValue, expectedOuts[i]);
                                cost += double.IsNaN(locCost) ? 0 : locCost;
                                break;
                            case LossType.Absolute:
                                outNeurLayer.neurons[i].CostValue = AbsoluteDerivative(outNeurLayer.neurons[i].OutValue, expectedOuts[i]) * Derivative.DeriveValue(outNeurLayer.neurons[i].InValue, outputActivation);
                                cost += Absolute(outNeurLayer.neurons[i].OutValue, expectedOuts[i]);
                                break;
                            default:
                                break;
                        }
                    }
                }
                else
                {
                    //To get the derived of InValues we need to make a separate [] because softmax needs them all
                    double[] derivedInValuesBySoftMax = new double[expectedOuts.Length];
                    for (int i = 0; i < derivedInValuesBySoftMax.Length; i++)
                        derivedInValuesBySoftMax[i] = outNeurLayer.neurons[i].InValue;

                    derivedInValuesBySoftMax = Derivative.DerivativeSoftMax(derivedInValuesBySoftMax);

                    for (int i = 0; i < outNeurLayer.neurons.Length; i++)
                    {
                        switch (loss)
                        {
                            case LossType.MeanSquare:
                                outNeurLayer.neurons[i].CostValue = MeanSquareDerivative(outNeurLayer.neurons[i].OutValue, expectedOuts[i]) * derivedInValuesBySoftMax[i];
                                cost += MeanSquare(outNeurLayer.neurons[i].OutValue, expectedOuts[i]);
                                break;
                            case LossType.CrossEntropy:
                                outNeurLayer.neurons[i].CostValue = CrossEntropyDerivative(outNeurLayer.neurons[i].OutValue, expectedOuts[i]) * derivedInValuesBySoftMax[i];
                                double locCost = CrossEntropy(outNeurLayer.neurons[i].OutValue, expectedOuts[i]);
                                cost += double.IsNaN(locCost) ? 0 : locCost;
                                break;
                            case LossType.Absolute:
                                outNeurLayer.neurons[i].CostValue = AbsoluteDerivative(outNeurLayer.neurons[i].OutValue, expectedOuts[i]) * derivedInValuesBySoftMax[i];
                                cost += Absolute(outNeurLayer.neurons[i].OutValue, expectedOuts[i]);
                                break;
                            default:
                                break;
                        }
                    }



                }
                return cost/expectedOuts.Length;
            }
            public static void CalculateLayerCost(NeuronLayer neuronLayer,WeightLayer connectionWeights, NeuronLayer nextNeuronLayer, ActivationType activation)
            {
                for (int i = 0; i < neuronLayer.neurons.Length; i++)
                {
                    double costVal = 0;
                    for (int j = 0; j < nextNeuronLayer.neurons.Length; j++)
                    {
                        costVal += nextNeuronLayer.neurons[j].CostValue * connectionWeights.weights[i][j];
                    }
                    costVal *= Derivative.DeriveValue(neuronLayer.neurons[i].InValue, activation);

                    neuronLayer.neurons[i].CostValue = costVal;
                }
            }

            private static double Absolute(double prediction, double label)
            {
                return Math.Abs(prediction - label);
            }
            private static double MeanSquare(double prediction, double label)
            {
                return (prediction - label) * (prediction - label);
            }
            private static double CrossEntropy(double prediction, double label)
            {
                return -label * Mathf.Log((float)prediction);
            }
            private static double AbsoluteDerivative(double prediction, double label)
            {
                if ((prediction - label) > 0)
                    return 1;
                return -1;
            }
            private static double MeanSquareDerivative(double prediction, double label)
            {
                return 2*(prediction - label);  
            }
            private static double CrossEntropyDerivative(double prediction, double label)
            {
                return -label / prediction;
            }
        }
    }
}