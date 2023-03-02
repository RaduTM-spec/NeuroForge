using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
using UnityEditor.ProjectWindowCallback;

public class NEATMoveRight : NEATAgent
{
    [Header("Attributes")]
    public float speed = 5f;
    public Transform target;
    Rigidbody rb;

    protected override void Awake()
    {
        base.Awake();
        rb = GetComponent<Rigidbody>();
    }
    protected void Update()
    {
        AddReward(Time.deltaTime);
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        transform.position += new Vector3(actionBuffer.ContinuousActions[0], 0, actionBuffer.ContinuousActions[1]) * Time.fixedDeltaTime * speed;
        /*switch (actionBuffer.DiscreteActions[0])
        {
            case 0:
                transform.position += Vector3.left * Time.fixedDeltaTime * speed;
                break;
            case 1:
                transform.position += Vector3.forward * Time.fixedDeltaTime * speed;
                break;
            case 2:
                transform.position += Vector3.right * Time.fixedDeltaTime * speed;
                break;
            case 3:
                transform.position += Vector3.back * Time.fixedDeltaTime * speed;
                break;
            case 4:
                //do nothing
                break;
        }*/
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        if(Input.GetKey(KeyCode.A))
        {
            actionSet.DiscreteActions[0] = 0;
        }
        else if(Input.GetKey(KeyCode.W))
        {
            actionSet.DiscreteActions[0] = 1;
        }
        else if(Input.GetKey(KeyCode.D))
        {
            actionSet.DiscreteActions[0] = 2;
        }
        else if(Input.GetKey(KeyCode.S))
        {
            actionSet.DiscreteActions[0] = 3;
        }
        else
        {
            actionSet.DiscreteActions[0] = 4;
        }
    }
    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name == "Target")
        {
            SetReward(transform.position.z + 100f);
            EndEpisode();
        }
        else if (collision.collider.CompareTag("Wall"))
        {
            SetReward(transform.position.z - 5f);
            EndEpisode();
        }

    }
    public void OnTriggerEnter(Collider other)
    {
        if(other.CompareTag("RAY"))
        {
            SetReward(transform.position.z);
            EndEpisode();
        }
    }
}
