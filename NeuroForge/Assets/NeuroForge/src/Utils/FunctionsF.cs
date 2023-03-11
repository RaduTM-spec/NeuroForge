using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NeuroForge
{
    public readonly struct FunctionsF
    {
        public static float RandomGaussian(float mean = 0, float standardDeviation = 1)
        {
            System.Random rng = new System.Random();
            double x1 = 1 - rng.NextDouble(); //zero exlusion anti log(0)
            double x2 = 1 - rng.NextDouble();

            float y1 = (float)(Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2));
            return y1 * standardDeviation + mean;
        }
        /// <summary>
        /// Returns a value in range [0.0, 1.0] (inclusive)
        /// </summary>
        /// <returns></returns>
        public static float RandomValue() => UnityEngine.Random.value;
        public static void Normalize(List<float> list)
        {
            // Calculate mean
            float mean = list.Average();

            // Calculate std
            float sum = 0f;
            foreach (var item in list)
            {
                sum += (item - mean) * (item - mean);
            }
            float variance = sum / list.Count;
            float std = MathF.Sqrt(variance);

            // Normalize list
            list = list.Select(x => (x - mean) / (std + 1e-8f)).ToList();
        }
        public readonly struct Activation
        {       
            public static float Activate(float value, ActivationTypeF activationType)
            {
                switch(activationType)
                {
                    case ActivationTypeF.Linear:
                        return Linear(value);
                    case ActivationTypeF.ModifiedSigmoid:
                        return ModifiedSigmoid(value);
                    case ActivationTypeF.HyperbolicTangent:
                        return HyperbolicTangent(value);
                    case ActivationTypeF.Sigmoid:
                        return Sigmoid(value);
                    case ActivationTypeF.Absolute:
                        return Absolute(value);
                    case ActivationTypeF.Inverse:
                        return Inverse(value);
                    case ActivationTypeF.Square:
                        return Square(value);
                    case ActivationTypeF.Sine:
                        return Sine(value);
                    case ActivationTypeF.Cosine:
                        return Cosine(value);
                    //case ActivationTypeF.BinaryStep:
                    //    return BinaryStep(value);
                    case ActivationTypeF.Reluctant:
                        return Reluctant(value);
                    case ActivationTypeF.Gaussian:
                        return Gaussian(value);                  
                    default:
                        throw new Exception("Unhandled activation type");
                }
            }

            public static float BinaryStep(float value) => value >= 0 ? 1 : 0;
            public static float Sigmoid(float value) => 1f / (1f + Mathf.Exp(-value));
            public static float ModifiedSigmoid(float value) => 1f / (1f + Mathf.Exp(-4.9f * value));
            public static float HyperbolicTangent(float value) => (float)Math.Tanh(value);
            public static float Linear(float value) => value;
            public static float Inverse(float value) => -value;
            public static float Square(float value) => value * value;
            public static float Sine(float value) => Mathf.Sin(value);
            public static float Cosine(float value) => Mathf.Cos(value);
            public static float Absolute(float value) => Mathf.Abs(value);
            public static float Reluctant(float value) => value > 0 ? 1f : 0f;
            public static float Gaussian(float value) => Mathf.Exp(-value * value / 2);
            public static void SoftMax(float[] values)
            {
                float exp_sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = MathF.Exp(values[i]);
                    exp_sum += values[i];
                }

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] /= exp_sum;
                }
            }
            public static int ArgMax(float[] values)
            {
                int index = -1;
                float max = float.MinValue;
                for (int i = 0; i < values.Length; i++)
                    if (values[i] > max)
                    {
                        max = values[i];
                        index = i;
                    }
                return index;
            }
        }
    }

    public enum ActivationTypeF
    {  
        Linear,
        ModifiedSigmoid,
        HyperbolicTangent,
        Sigmoid,
        Absolute,
        Inverse,
        Square,
        Sine,
        Cosine,      
        //BinaryStep,   
        Reluctant,
        Gaussian,
    }
}
