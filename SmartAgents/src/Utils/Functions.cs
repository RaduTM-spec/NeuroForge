using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SmartAgents
{
    public struct Functions
    {
        public static double RandomGaussian(double mean = 0, double standardDeviation = 1.0)
        {         
            System.Random rng = new System.Random();
            double x1 = 1 - rng.NextDouble();
            double x2 = 1 - rng.NextDouble();

            double y1 = Math.Sqrt(-2.0f * Math.Log(x1)) * Math.Cos(2.0f * Math.PI * x2);
            return y1 * standardDeviation + mean;          
        }
        public static void PrintArray(Array array)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[ ");
            foreach (var item in array)
            {
                sb.Append(item.ToString());
                sb.Append(", ");
            }
            sb.Remove(sb.Length - 2, 1);
            sb.Append("]");
            Debug.Log(sb.ToString());
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
            public static void ActivateOutputLayer(NeuronLayer outNeurLayer, ActivationType outActivationFunction)
            {
                if (outActivationFunction == ActivationType.SoftMax)
                {
                    double[] InValuesToActivate = outNeurLayer.neurons.Select(x => x.InValue).ToArray();
                    Activation.SoftMax(InValuesToActivate);
                    for (int i = 0; i < InValuesToActivate.Length; i++)
                    {
                        outNeurLayer.neurons[i].OutValue = InValuesToActivate[i];
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
            internal static double ActivateValue(double value, ActivationType activationFunction)
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
            public static double Sigmoid(double value) => 1 / (1 + Math.Exp(-value));
            public static double Tanh(double value) => Math.Tanh(value);
            public static double ReLU(double value) => Math.Max(0, value);
            public static double LeakyReLU(double value, double alpha = 0.2) => value > 0 ? value : value * alpha;
            public static double SiLU(double value) => value * Sigmoid(value);
            public static double SoftPlus(double value) => Math.Log(1 + Math.Exp(value));
            public static void SoftMax(double[] values)
            {
                double exp_sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Math.Exp(values[i]);
                    exp_sum += values[i];
                }

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] /= exp_sum;
                }
            }
            public static int ArgMax(double[] values)
            {
                int index = -1;
                double max = double.MinValue;
                for (int i = 0; i < values.Length; i++)
                    if (values[i] > max)
                    {
                        max = values[i];
                        index = i;
                    }
                return index;
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
            public static void DeriveOutputLayer(NeuronLayer outNeurLayer, ActivationType outActivationFunc)
            {
                if (outActivationFunc == ActivationType.SoftMax)
                {
                    double[] InValuesToDerive = outNeurLayer.neurons.Select(x => x.OutValue).ToArray();
                    Derivative.DerivativeSoftMax(InValuesToDerive);
                    for (int i = 0; i < InValuesToDerive.Length; i++)
                    {
                        outNeurLayer.neurons[i].CostValue = InValuesToDerive[i];
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
            static internal double DeriveValue(double value, ActivationType activationFunction)
            {
                switch (activationFunction)
                {
                    case ActivationType.Tanh:
                        return DerivativeTanh(value);
                    case ActivationType.BinaryStep:
                        return DerivativeBinaryStep();
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

            static public double DerivativeTanh(double  value) => 1f - Math.Pow(Math.Tanh(value), 2);
            static public double DerivativeSigmoid(double value) => Activation.Sigmoid(value) * (1 - Activation.Sigmoid(value));
            static public double DerivativeBinaryStep() => 0;
            static public double DerivativeReLU(double value) => value < 0 ? 0 : 1;
            static public double DerivativeLeakyReLU(double value, double alpha = 0.2f) => value < 0 ? alpha : 1;
            static public double DerivativeSiLU(double value) => (1 + Math.Exp(-value) + value * Math.Exp(-value)) / Math.Pow(1 + Math.Exp(-value), 2);
            static public double DerivativeSoftPlus(double value) => Activation.Sigmoid(value);
            static public void DerivativeSoftMax(double[] values)
            {
                double exp_sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Math.Exp(values[i]);
                    exp_sum += values[i];
                }

                double squared_sum = exp_sum*exp_sum;

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = values[i] * exp_sum - values[i] * values[i] / squared_sum;
                }
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
                    double[] derivedInValuesBySoftMax = new double[expectedOuts.Length];
                    for (int i = 0; i < derivedInValuesBySoftMax.Length; i++)
                        derivedInValuesBySoftMax[i] = outNeurLayer.neurons[i].InValue;

                    Derivative.DerivativeSoftMax(derivedInValuesBySoftMax);

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