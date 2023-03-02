using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoleBalancer : NEATAgent
{
    public float rotationSpeed = 1f;
    public Transform rotator;
    public Rigidbody rotatorRB;
    public HingeJoint pole;

    public void Update()
    {
        AddReward(Time.deltaTime);
    }

    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        //5 obs
        sensorBuffer.AddObservation(pole.angle);
        sensorBuffer.AddObservation(rotatorRB.angularVelocity.normalized);
        sensorBuffer.AddObservation(rotatorRB.angularDrag);
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        rotator.Rotate(Vector3.forward, rotationSpeed * actionBuffer.ContinuousActions[0]);
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        if(Input.GetKey(KeyCode.A))
        {
            actionSet.ContinuousActions[0] = 1;
        }
        else if(Input.GetKey(KeyCode.D))
        {
            actionSet.ContinuousActions[0] = -1;
        }
        
    }
}
