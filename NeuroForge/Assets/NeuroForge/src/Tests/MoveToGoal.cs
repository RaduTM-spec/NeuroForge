using NeuroForge;
using UnityEngine;

public class MoveToGoal : PPOAgent
{
    [Header("Attributes")]
    public float speed = 5f;
    public Transform target;

    public Vector2 x_range;
    public Vector2 z_range;

    public ActionType type = ActionType.Continuous;

    protected override void Awake()
    {
        base.Awake();
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        //4 inputs
        float x_pos = transform.localPosition.x / 6f;
        float z_pos = transform.localPosition.z / 5f;

        float t_x_pos = target.localPosition.x;
        float t_z_pos = target.localPosition.z;

        sensorBuffer.AddObservation(x_pos);
        sensorBuffer.AddObservation(z_pos);
        sensorBuffer.AddObservation(t_x_pos);
        sensorBuffer.AddObservation(t_z_pos);
    }
    public override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        if(type == ActionType.Continuous)
        {
            transform.position +=
             new Vector3(actionBuffer.ContinuousActions[0], 0, actionBuffer.ContinuousActions[1])
             * Time.fixedDeltaTime
             * speed;
        }
       else if(type == ActionType.Discrete)
       {
            switch (actionBuffer.DiscreteActions[0])
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
            switch (actionBuffer.DiscreteActions[1])
            {
                case 0:
                    break;
                case 1:
                    break;

            }
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
