using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class NEATXORTest : NEATAgent
{
    [Space]
    public int input1 = -1;
    public int input2 = -1;
    public int XOR = -1;

    int oneGenerationTests = 4;

    protected override void Awake()
    {
        base.Awake();
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        //2 inputs
        input1 = XORInputGenerator.inputs[0];
        input2 = XORInputGenerator.inputs[1];
        sensorBuffer.AddObservation(input1);
        sensorBuffer.AddObservation(input2);
    }


    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        XOR = input1 ^ input2;

        int prediction = actionBuffer.DiscreteActions[0];
        if (prediction == XOR)
            AddReward(1);

        if(--oneGenerationTests == 0)
        {
            oneGenerationTests = 4;
            EndEpisode();
        }
    }
}
