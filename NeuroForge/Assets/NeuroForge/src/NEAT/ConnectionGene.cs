using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class ConnectionGene : ICloneable
    {
        // Never use get; set on fields that are serializable
        [SerializeField] public int innovation;

        [SerializeField] public float weight;
        [SerializeField] public bool enabled;

        [SerializeField] public int inNeuron;
        [SerializeField] public int outNeuron;


        public ConnectionGene(NodeGene inNeuron, NodeGene outNeuron, int innovation)
        {
            this.innovation = innovation;
            this.inNeuron = inNeuron.innovation;
            this.outNeuron = outNeuron.innovation;
            this.enabled = true;
            this.weight = FunctionsF.RandomValue() < 0.5f ?
                          FunctionsF.RandomGaussian(0, 0.1f) :
                          FunctionsF.RandomGaussian(1, 0.1f);
            outNeuron.incomingConnections.Add(this.innovation);
        }
        private ConnectionGene() { }
        public object Clone()
        {
            ConnectionGene clone = new ConnectionGene();
            
            clone.innovation = this.innovation;
            clone.weight = this.weight;
            clone.enabled = this.enabled;
            clone.inNeuron = this.inNeuron;
            clone.outNeuron = this.outNeuron;
        
            return clone;
        }
        public bool IsSequencial() => inNeuron == outNeuron;
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[ ");
            sb.Append(innovation);
            sb.Append(", in: ");
            sb.Append(inNeuron);
            sb.Append(", out: ");
            sb.Append(outNeuron);
            sb.Append(" ]");
            return sb.ToString();
        }
    }
}
