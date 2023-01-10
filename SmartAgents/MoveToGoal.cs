using SmartAgents;
using UnityEngine;

public class MoveToGoal : Agent
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
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(transform.localPosition);
        sensorBuffer.AddObservation(target.localPosition);
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        transform.position += new Vector3(actionBuffer.continuousActions[0], 0, actionBuffer.continuousActions[1]) * Time.deltaTime * speed;
        
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
            AddReward(11);
            EndEpisode();
        }
        else if (collision.collider.CompareTag("Wall"))
        { 
            AddReward(-11);
            EndEpisode();
        }

    }

}
