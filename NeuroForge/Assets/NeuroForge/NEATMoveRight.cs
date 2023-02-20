using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
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
    protected override void Update()
    {
        base.Update();
        AddReward(transform.position.x);
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {

    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        //transform.position += new Vector3(actionBuffer.continuousActions[0], 0, actionBuffer.continuousActions[1]) * Time.deltaTime * speed;
        switch (actionBuffer.discreteActions[0])
        {
            case 0:
                transform.position += Vector3.left * Time.deltaTime * speed;
                break;
            case 1:
                transform.position += Vector3.forward * Time.deltaTime * speed;
                break;
            case 2:
                transform.position += Vector3.right * Time.deltaTime * speed;
                break;
            case 3:
                transform.position += Vector3.back * Time.deltaTime * speed;
                break;
            default:
                //do nothing
                break;
        }
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        actionSet.continuousActions[0] = Input.GetAxisRaw("Horizontal");
        actionSet.continuousActions[1] = Input.GetAxisRaw("Vertical");
    }
    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name == "Target")
        {
            EndEpisode();
        }
        else if (collision.collider.CompareTag("Wall"))
        {
            EndEpisode();
        }

    }
}
