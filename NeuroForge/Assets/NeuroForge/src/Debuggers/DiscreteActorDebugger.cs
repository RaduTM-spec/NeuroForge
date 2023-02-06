using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DiscreteActorDebugger : MonoBehaviour
{
    public PPOActorNetwork discreteNet;

    private void Update()
    {
        double[] inputs = new double[] {transform.position.x, transform.position.y, transform.position.z, transform.position.x, transform.position.y, transform.position.z };
        double[] probs = discreteNet.DiscreteForwardPropagation(inputs).Item1;
        Functions.Print(probs);
    }
}
