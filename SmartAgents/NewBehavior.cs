using SmartAgents;
using UnityEngine;

public class NewBehavior : Agent
{
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(new double[] {5,3,2,4});
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        Debug.Log(actionBuffer.ToString());
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        actionSet.actions[0] = 0.4444456;
        actionSet.actions[1] = -0.98;
        actionSet.actions[2] = 0.55;
        actionSet.actions[3] = 0.32;
        AddReward(54);
    }
    
}
