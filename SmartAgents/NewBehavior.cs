using SmartAgents;
using UnityEngine;

public class NewBehavior : Agent
{
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(1);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        Debug.Log(actionBuffer.ToString());
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        
    }

}
