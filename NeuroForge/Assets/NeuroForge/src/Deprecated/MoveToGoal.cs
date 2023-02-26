using NeuroForge;
using UnityEngine;

public class MoveToGoal : PPOAgent
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
        //6 inputs
        sensorBuffer.AddObservation(transform.localPosition);
        sensorBuffer.AddObservation(target.localPosition);
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        //transform.position += new Vector3(actionBuffer.continuousActions[0], 0, actionBuffer.continuousActions[1]) * Time.deltaTime * speed;
       switch(actionBuffer.DiscreteActions[0])
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
        }
       switch(actionBuffer.DiscreteActions[1])
        {
            case 0:
                break;
            case 1:               
                break;

        }
        
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        actionSet.ContinuousActions[0] = Input.GetAxisRaw("Horizontal");
        actionSet.ContinuousActions[1] = Input.GetAxisRaw("Vertical");
    }
    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name == "Target")
        {
            AddReward(+1);
            EndEpisode();
        }
        else if (collision.collider.CompareTag("Wall"))
        { 
            AddReward(-1);
            EndEpisode();
        }

    }

}
