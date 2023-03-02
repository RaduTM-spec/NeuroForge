using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
using System.Linq;

public class NEATTest : NEATAgent
{
    public float input;
    public int roundPerGeneration = 10;
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {

        sensorBuffer.AddObservation(FunctionsF.RandomValue());
        sensorBuffer.AddObservation(FunctionsF.RandomValue());
        sensorBuffer.AddObservation(FunctionsF.RandomValue());
        sensorBuffer.AddObservation(FunctionsF.RandomValue());
        sensorBuffer.AddObservation(FunctionsF.RandomValue());

        /*input = FunctionsF.RandomGaussian(0, 1.5f);
        sensorBuffer.AddObservation(input);*/
    }


    public override void OnActionReceived(in ActionBuffer actionBuffer){        // Calculate the reward based on the action buffer

        AddReward(actionBuffer.ContinuousActions[0]);
        AddReward(actionBuffer.ContinuousActions[1]);
        AddReward(actionBuffer.ContinuousActions[2]);
        AddReward(actionBuffer.ContinuousActions[3]);
        AddReward(actionBuffer.ContinuousActions[4]);

        EndEpisode();
        /*// Calculate the reward based on inputs
        float sinOf = Mathf.Sin(input);
        float error = Mathf.Abs(actionBuffer.ContinuousActions[0] - sinOf);
        float reward = 1f / error;
        AddReward(reward);

        roundPerGeneration--;
        if(roundPerGeneration == 0)
        {
            roundPerGeneration = 10;
            EndEpisode();
        }*/

    }
}
