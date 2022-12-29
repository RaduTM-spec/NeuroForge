using System;
using UnityEngine;
namespace SmartAgents
{
    [Serializable]
    class Sample
    {
        [SerializeField] public double[] state;
        [SerializeField] public double[] action;
        [SerializeField] public float reward;
        [SerializeField] public double[] nextState;
    }
}
