using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using UnityEngine;

namespace NeuroForge
{
    public struct Functions
    {
        public static double RandomGaussian(double mean = 0, double stddev = 1)
        {         
            System.Random rng = new System.Random();
            double x1 = 1 - rng.NextDouble();
            double x2 = 1 - rng.NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * stddev + mean;          
        }
        public static double RandomValue() => UnityEngine.Random.value;
        public static T RandomIn<T>(IEnumerable<T> values)
        {
            int randIndex = UnityEngine.Random.Range(0, values.Count());
            try
            {
                
                return values.ElementAt(randIndex);

            }
            catch
            {
                Debug.LogError("IEnumrable is empty");
                return values.First();
            }
            
        }
        public static T RandomIn<T>(IEnumerable<T> values, List<float> probabilities)
        {           
            for (int i = 0; i < probabilities.Count; i++)
            {
                probabilities[i] = MathF.Exp(probabilities[i]);
            }
            float sum = probabilities.Sum();
            float random = FunctionsF.RandomValue() * sum;
            int index = 0;
            while (random > probabilities[index])
            {
                random -= probabilities[index];
                index++;
            }
            return values.ElementAt(index);   
        }
        public static void Normalize(List<double> list)
        {
            // Calculate mean
            double mean = list.Average();

            // Calculate std
            double sum = 0;
            foreach (var item in list)
            {
                sum += (item - mean) * (item - mean);
            }
            double std = Math.Sqrt(sum / list.Count);

            // Normalize list
            for (int i = 0; i < list.Count; i++)
            {
                list[i] = (list[i] - mean) / (std + 1e-8);
            }
        }
        public static void Shuffle<T>(List<T> list, int iterations = 1)
        {
            var random = new System.Random();
            for(int iter = 0; iter < iterations; iter++)
            {
                for (int i = 0; i < list.Count; i++)
                {
                    int j = random.Next(0, list.Count - 1);
                    T temp = list[i];
                    list[i] = list[j];
                    list[j] = temp;
                }
            }
           
        }
        public static void Print(IEnumerable array, string tag = null)
        {
            StringBuilder sb = new StringBuilder();
            if (tag != null) sb.Append(tag);
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
        public static string StringOf(IEnumerable array, string tag = null)
        {
            StringBuilder sb = new StringBuilder();
            if (tag != null) sb.Append(tag);
            sb.Append("[ ");
            foreach (var item in array)
            {
                sb.Append(item.ToString());
                sb.Append(", ");
            }
            sb.Remove(sb.Length - 2, 1);
            sb.Append("]");
            return sb.ToString();
        }
        public static bool IsValueIn<T>(T value, IEnumerable<T> collection)
        {
            foreach (var item in collection)
            {
                if(value.Equals(item)) return true;
            }
            return false;
        }
        public static void DebugInFile(string text, bool newLine = true)
        {
            using (StreamWriter sw = new StreamWriter("C:\\Users\\X\\Desktop\\debug.txt", true))
            {
                if (newLine)
                    sw.WriteLine(text);
                else
                    sw.Write(text);
            }
        }
        public static string HexOf(Color color)
        {
            int r = Mathf.RoundToInt(color.r * 255.0f);
            int g = Mathf.RoundToInt(color.g * 255.0f);
            int b = Mathf.RoundToInt(color.b * 255.0f);

            return string.Format("#{0:X2}{1:X2}{2:X2}", r, g, b);
        }
        public static bool HasNaN(IEnumerable<double> array)
        {
            foreach (var item in array)
            {
                if (double.IsNaN(item)) return true;
            }
            return false;
        }

        public readonly struct Activation
        {
            internal static double ActivateValue(double value, ActivationType activationFunction)
            {
                switch(activationFunction)
                {
                    case ActivationType.Tanh:
                        return TanH(value);
                    case ActivationType.Sigmoid:
                        return Sigmoid(value);
                    case ActivationType.Relu:
                        return ReLU(value);
                    case ActivationType.LeakyRelu:
                        return LeakyReLU(value);
                    case ActivationType.Silu:
                        return SiLU(value);
                    case ActivationType.ELU:
                        return ELU(value);
                    case ActivationType.SoftPlus:
                        return SoftPlus(value);
                    default: //Linear
                        return value;
                        
                }
                
            }

            public static double Sigmoid(double value) => 1.0 / (1.0 + Math.Exp(-value)); 
            public static double TanH(double value) => Math.Tanh(value);
            public static double ReLU(double value) => Math.Max(0, value);
            public static double LeakyReLU(double value, double alpha = 0.2) => value > 0 ? value : value * alpha;
            public static double SiLU(double value) => value * Sigmoid(value);
            public static double SoftPlus(double value) => Math.Log(1 + Math.Exp(value));
            public static double ELU(double value) => value < 0? Math.Exp(value) - 1 : value;
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
            public static void OneHot(double[] values)
            {
                int index = ArgMax(values);
                for (int i = 0; i < values.Length; i++)
                {
                    if (i == index)
                        values[i] = 1;
                    else
                        values[i] = 0;
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
        public readonly struct Derivative
        {
            static internal double DeriveValue(double value, ActivationType activationFunction)
            {
                switch (activationFunction)
                {
                    case ActivationType.Tanh:
                        return TanH(value);
                    case ActivationType.Sigmoid:
                        return Sigmoid(value);
                    case ActivationType.Relu:
                        return ReLU(value);
                    case ActivationType.LeakyRelu:
                        return LeakyReLU(value);
                    case ActivationType.Silu:
                        return SiLU(value);
                    case ActivationType.ELU:
                        return ELU(value);
                    case ActivationType.SoftPlus:
                        return SoftPlus(value);
                    default: // Linear
                        return 1;
                }
            }
            
            static public double TanH(double  value)
            {
                double e2 = Math.Exp(2 * value);
                double tanh = (e2 - 1) / (e2 + 1);
                return 1 - tanh * tanh;
                
            }
            static public double Sigmoid(double value)
            {
                double act = Activation.Sigmoid(value);
                return act * (1 - act);
            }
            static public double ReLU(double value) => value > 0 ? 1 : 0;
            static public double LeakyReLU(double value, double alpha = 0.2) => value > 0 ? 1 : alpha;
            static public double SiLU(double value)
            {
                double sigm = Activation.Sigmoid(value);
                return value * sigm * (1 - sigm) + sigm;
            }
            static public double ELU(double value) => value < 0 ? Math.Exp(value) : 1;
            static public double SoftPlus(double value) => Activation.Sigmoid(value);
            static public void SoftMax(double[] values)
            {
                double exp_sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Math.Exp(values[i]);
                    exp_sum += values[i];
                }

                double squared_sum = exp_sum * exp_sum;

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (values[i] * exp_sum - values[i] * values[i]) / squared_sum;
                }
            }
        }
        public readonly struct Cost
        {
            public static double Absolute(double prediction, double label)
            {
                return Math.Abs(prediction - label);
            }
            public static double MeanSquare(double prediction, double label)
            {
                return (prediction - label) * (prediction - label);
            }
            public static double CrossEntropy(double prediction, double label)
            {
                return -label * Math.Log(prediction);
            }

            public static double AbsoluteDerivative(double prediction, double label)
            {
                if ((prediction - label) > 0)
                    return 1;
                return -1;
            }
            public static double MeanSquareDerivative(double prediction, double label)
            {
                return prediction - label;
            }
            public static double CrossEntropyDerivative(double prediction, double label)
            {
                prediction += 1e-10;
                return (-prediction + label) / (prediction * (prediction - 1));
            }
        }     

    }
}