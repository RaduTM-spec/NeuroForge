using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class ConnectionGene
    {
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

            weight = Functions.RandomValue() < 0.5f ?
                     FunctionsF.RandomGaussian(0, 0.1f) :
                     FunctionsF.RandomGaussian(1, 0.1f);
            outNeuron.incomingConnections.Add(this.innovation);
        }
    }
}
