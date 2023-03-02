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
    public int result = -1;

    public int generationSteps = 10;
    int oneGenerationTests = 10;

    protected override void Awake()
    {
        base.Awake();
        oneGenerationTests = generationSteps;
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
        result = input1 ^ input2;

        // 2 discrete outputs
        // o1 is for 0
        // o2 is for 1

        int prediction = actionBuffer.DiscreteActions[0];
        if (prediction == result)
            AddReward(1);

        if(--oneGenerationTests == 0)
        {
            oneGenerationTests = generationSteps;
            EndEpisode();
        }
    }
}
