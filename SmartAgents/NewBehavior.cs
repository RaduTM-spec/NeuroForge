using SmartAgents;
using UnityEngine;

public class NewBehavior : Agent
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
        sensorBuffer.AddObservation(transform.position);
        sensorBuffer.AddObservation(target.position);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        transform.position += new Vector3((float)actionBuffer.actions[0], 0, (float)actionBuffer.actions[1]) * Time.deltaTime * speed;
    }
    public override void Heuristic(ActionBuffer actionSet)
    {
        actionSet.actions[0] = Input.GetAxis("Horizontal");
        actionSet.actions[1] = Input.GetAxis("Vertical");
    }
    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name == "Target")
        {
            AddReward(1);
            EndAction();
        }
        else if (collision.collider.CompareTag("Wall"))
        { 
            AddReward(-1);
            EndAction();
        }

    }

}
