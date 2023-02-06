using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class NodeGene
    {
        [SerializeField] public int innovation;
        [SerializeField] public float InValue;
        [SerializeField] public float OutValue;

        [SerializeField] public ActivationTypeF activationType;
        [SerializeField] public NEATNodeType type;
        [SerializeField] public List<int> incomingConnections;

        public NodeGene(int innov, NEATNodeType type)
        {
            this.innovation = innov;
            InValue = 0;
            OutValue = type == NEATNodeType.bias ? 1 : 0;

            this.type = type;
            activationType = (ActivationTypeF)(int)(FunctionsF.RandomValue() * Enum.GetValues(typeof(ActivationTypeF)).Length);
            incomingConnections = new List<int>();
        }
        public void Activate()
        {
            OutValue = FunctionsF.Activation.Activate(InValue, activationType);
        }

    }
    public enum NEATNodeType
    {
        input,
        hidden,
        output,
        bias,
    }
}
