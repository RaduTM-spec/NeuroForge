using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
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
            public static void ActivateLayer(NeuronLayer neuronLayer, ActivationType activationFunction)
            {               
                foreach (Neuron neuron in neuronLayer.neurons)
                {
                    neuron.OutValue = ActivateValue(neuron.InValue, activationFunction);
                }
            }
            public static void ActivateOutputLayer(NeuronLayer outNeurLayer, ActivationType outActivationFunction, int[] outputFormat)
            {
                if (outActivationFunction == ActivationType.BranchedSoftMaxActivation)
                {
                    // Get all raw values
                    List<double> rawValues = outNeurLayer.neurons.Select(x => x.InValue).ToList();

                    int index = 0;

                    // Foreach branch, activate the branch values
                    foreach (var branch in outputFormat)
                    {
                        // Get the branch from raw values
                        double[] branchValues = rawValues.GetRange(index, index + branch).ToArray();

                        // Activate the branch
                        SoftMax(branchValues);

                        // Place the activated branch on OutValues
                        for (int i = index; i < index + branch; i++)
                        {
                            outNeurLayer.neurons[i].OutValue = branchValues[i - index];
                        }

                        index += branch;
                    }


                }
                else if (outActivationFunction == ActivationType.PairedTanhSoftPlusActivation)
                {
                    for (int i = 0; i < outNeurLayer.neurons.Length; i++)
                    {
                        outNeurLayer.neurons[i].OutValue =
                            i % 2 == 0 ?
                            ActivateValue(outNeurLayer.neurons[i].InValue, ActivationType.Tanh) :     // mu
                            ActivateValue(outNeurLayer.neurons[i].InValue, ActivationType.SoftPlus);  // sigma
                    }
                }
                else
                {
                    foreach (Neuron neuron in outNeurLayer.neurons)
                    {
                        neuron.OutValue = ActivateValue(neuron.InValue, outActivationFunction);
                    }
                }
            }
            public static double ActivateValue(double value, ActivationType activationFunction)
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
                    case ActivationType.SoftPlus:
                        return SoftPlus(value);
                    default:
                        return value;
                        
                }
                
            }

            public static double BinaryStep(double value) => value < 0 ? 0 : 1;
            public static double Sigmoid(double value) => 1 / (1 + Mathf.Exp((float)-value));
            public static double Tanh(double value) => Math.Tanh(value);
            public static double ReLU(double value) => Mathf.Max(0, (float) value);
            public static double LeakyReLU(double value, double alpha = 0.2) => value > 0 ? value : value * alpha;
            public static double SiLU(double value) => value * Sigmoid(value);
            public static double SoftPlus(double value) => Math.Log(1 + Math.Exp(value));
            public static double[] SoftMax(ICollection<double> values)
            {
                double[] returns = new double[values.Count];
                values.CopyTo(returns, 0);
                double sum = 0;
                for (int i = 0; i < values.Count; i++)
                {
                    returns[i] = Mathf.Exp((float)returns[i]);
                    sum += returns[i];
                }
                return returns;
            }

        }
        public struct Derivative
        {
            public static void DeriveLayer(NeuronLayer neuronLayer, ActivationType activationFunction)
            {         
                foreach (Neuron neuron in neuronLayer.neurons)
                {
                    neuron.CostValue = DeriveValue(neuron.OutValue, activationFunction);
                }
                
            }
            public static void DeriveOutputLayer(NeuronLayer outNeurLayer, ActivationType outActivationFunc, int[] outputFormat)
            {
                if (outActivationFunc == ActivationType.BranchedSoftMaxActivation)
                {
                    // Get all out values
                    List<double> rawValues = outNeurLayer.neurons.Select(x => x.OutValue).ToList();

                    int index = 0;

                    // Foreach branch, derive the out branch values
                    foreach (var branch in outputFormat)
                    {
                        // Get the branch from activated values
                        double[] branchValues = rawValues.GetRange(index, index + branch).ToArray();

                        // Derive the branch values
                        DerivativeSoftMax(branchValues);

                        // Place the derived branch values in CostValues
                        for (int i = index; i < index + branch; i++)
                        {
                            outNeurLayer.neurons[i].CostValue = branchValues[i - index];
                        }

                        index += branch;
                    }
                }
                else if (outActivationFunc == ActivationType.PairedTanhSoftPlusActivation)
                {
                    for (int i = 0; i < outNeurLayer.neurons.Length; i++)
                    {
                        outNeurLayer.neurons[i].CostValue =
                            i % 2 == 0 ?
                            DeriveValue(outNeurLayer.neurons[i].OutValue, ActivationType.Tanh) :     // mu
                            DeriveValue(outNeurLayer.neurons[i].OutValue, ActivationType.SoftPlus);  // sigma
                    }
                }
                else
                {
                    foreach (Neuron neuron in outNeurLayer.neurons)
                    {
                        neuron.CostValue = DeriveValue(neuron.OutValue, outActivationFunc);
                    }
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
                    case ActivationType.SoftPlus:
                        return DerivativeSoftPlus(value);
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
            static public double DerivativeSoftPlus(double value)
            {
                return Activation.Sigmoid(value);
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
                            case LossType.MeanAbsolute:
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
                            case LossType.MeanAbsolute:
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
                return -label * Math.Log(prediction);
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
                prediction += 0.0000000001;
                return (-prediction + label) / (prediction * (prediction - 1));
            }
        }
    }
}