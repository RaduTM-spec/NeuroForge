using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
public class NEATTest : NEATAgent
{
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(Functions.RandomValue());
        sensorBuffer.AddObservation(Functions.RandomValue());
        sensorBuffer.AddObservation(Functions.RandomValue());
        sensorBuffer.AddObservation(Functions.RandomValue());
        sensorBuffer.AddObservation(Functions.RandomValue());
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        AddReward(actionBuffer.ContinuousActions[0]);
        AddReward(actionBuffer.ContinuousActions[1]);
        AddReward(actionBuffer.ContinuousActions[2]);
        AddReward(actionBuffer.ContinuousActions[3]);
        AddReward(actionBuffer.ContinuousActions[4]);
        EndEpisode();
    }
}
