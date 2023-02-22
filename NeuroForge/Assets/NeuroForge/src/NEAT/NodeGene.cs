using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
        [SerializeField] public float layer;

        public NodeGene(int innov, NEATNodeType type, float layer)
        {
            this.innovation = innov;
            InValue = 0;
            OutValue = type == NEATNodeType.bias ? 1 : 0;

            this.type = type;

            activationType = type == NEATNodeType.hidden? 
                            (ActivationTypeF)(int)(FunctionsF.RandomValue() * Enum.GetValues(typeof(ActivationTypeF)).Length) :     
                            ActivationTypeF.Linear;

            incomingConnections = new List<int>();
            this.layer = layer;
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
            clone.layer = this.layer;
            return clone;
        }
        public void Activate()
        {
            OutValue = FunctionsF.Activation.Activate(InValue, activationType);
            activated = true;
        }
        public void Deactivate() => activated = false;
        public bool IsActivated() => activated;
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[ ");
            sb.Append(innovation);
            sb.Append(", ");
            sb.Append(" in: ");
            foreach (var item in incomingConnections)
            {
                sb.Append(item);
                sb.Append(", ");
            }
            sb.Remove(sb.Length- 2, 1);
            sb.Append(']');
            return sb.ToString();
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
