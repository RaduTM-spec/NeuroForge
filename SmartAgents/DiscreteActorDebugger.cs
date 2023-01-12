using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DiscreteActorDebugger : MonoBehaviour
{
    public ActorNetwork discreteNet;

    private void Update()
    {
        double[] inputs = new double[] {transform.position.x, transform.position.y, transform.position.z, transform.position.x, transform.position.y, transform.position.z };
        double[] probs = discreteNet.DiscreteForwardPropagation(inputs).Item1;
        Functions.PrintArray(probs);
    }
}
