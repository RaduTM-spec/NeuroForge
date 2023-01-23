using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ForwardDebugger : MonoBehaviour
{
    public ActorNetwork actNet;

    private void Start()
    {
        double[] outs = actNet.DiscreteForwardPropagation(new double[] { 1,1,1,1,1,1}).Item1;
        Functions.Print(outs);
    }
}
