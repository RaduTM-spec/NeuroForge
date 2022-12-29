using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SmartAgents
{
    public struct Functions
    {
        public static double RandomGaussian(float mean = 0, float standardDeviation = 1)
        {
            
            System.Random rng = new System.Random();
            float x1 = (float)(1 - rng.NextDouble());
            float x2 = (float)(1 - rng.NextDouble());

            float y1 = Mathf.Sqrt(-2.0f * Mathf.Log(x1)) * Mathf.Cos(2.0f * (float)Math.PI * x2);
            return y1 * standardDeviation + mean;
            
        }
        public struct Activation
        {
            public static double Tanh(double value)
            {
                return Math.Tanh(value);
            }
        }
    }
}