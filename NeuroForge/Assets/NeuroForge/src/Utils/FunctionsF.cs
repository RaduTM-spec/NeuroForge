using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    public readonly struct FunctionsF
    {
        public static float RandomGaussian(float mean = 0, float standardDeviation = 1)
        {
            System.Random rng = new System.Random();
            double x1 = 1 - rng.NextDouble();
            double x2 = 1 - rng.NextDouble();

            float y1 = (float)(Math.Sqrt(-2 * Math.Log(x1)) * Math.Cos(2 * Math.PI * x2));
            return y1 * standardDeviation + mean;
        }
        public static float RandomValue() => UnityEngine.Random.value;
        public readonly struct Activation
        {       
            public static float Activate(float value, ActivationTypeF activationType)
            {
                switch(activationType)
                {
                    case ActivationTypeF.Linear:
                        return Linear(value);
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
                    case ActivationTypeF.Sigmoid:
                        return Sigmoid(value);
                    case ActivationTypeF.HyperbolicTangent:
                        return HyperbolicTangent(value);                            
                    case ActivationTypeF.Reluctant:
                        return Reluctant(value);
                    case ActivationTypeF.Gaussian:
                        return Gaussian(value);
                    default:
                        throw new Exception("Unhandled activation type");
                }
            }

            public static float Sigmoid(float value) => 1f / (1f + Mathf.Exp(-value));
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

        }
    }

    public enum ActivationTypeF
    {  
        Linear,
        Absolute,
        Inverse,
        Square,
        Sine,
        Cosine,
        Sigmoid,
        HyperbolicTangent,
        Reluctant,
        Gaussian,
    }
}
