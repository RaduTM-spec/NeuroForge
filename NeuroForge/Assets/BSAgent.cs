using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class BSAgent : NEATAgent
{
    public int iter = 100;
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(BinaryStringGenerator.string01);
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        int prediction = actionBuffer.DiscreteActions[0];

        //if is even pred = 0, if is odd pred = 1
        int correct = BinaryStringGenerator.string01.Sum() % 2;

        if (prediction == correct)
            AddReward(1);
        if(--iter == 0)
        {
            iter = 100;
            EndEpisode();
        }
    }
}
