using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class NodeGene : ICloneable
    {
        [SerializeField] public int innovation;
        [SerializeField] public float InValue;
        [SerializeField] public float OutValue;

        [SerializeField] public ActivationTypeF activationType;
        [SerializeField] public NEATNodeType type;
        [SerializeField] public List<int> incomingConnections;

        [SerializeField] private bool activated;

        public NodeGene(int innov, NEATNodeType type)
        {
            this.innovation = innov;
            InValue = 0;
            OutValue = type == NEATNodeType.bias ? 1 : 0;

            this.type = type;
            activationType = (ActivationTypeF)(int)(FunctionsF.RandomValue() * Enum.GetValues(typeof(ActivationTypeF)).Length);
            incomingConnections = new List<int>();

            activated = true;
        }
        private NodeGene() { }
        public object Clone()
        {
            NodeGene clone = new NodeGene();
            clone.innovation = innovation;
            clone.InValue= InValue;
            clone.OutValue= OutValue;
            clone.activationType = activationType;
            clone.type = type;
            clone.incomingConnections = this.incomingConnections.ToList();
            clone.activated = activated;
            return clone;
        }
        public void Activate()
        {
            OutValue = FunctionsF.Activation.Activate(InValue, activationType);
            activated = true;
        }
        public void Deactivate() => activated = false;
        public bool IsActivated() => activated;
    }
    public enum NEATNodeType
    {
        input,
        hidden,
        output,
        bias,
    }
}
