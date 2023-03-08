using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using System.IO;

namespace NeuroForge
{
    [DisallowMultipleComponent, RequireComponent(typeof(PPOHyperParameters))]
    public class PPOAgent : MonoBehaviour
    {
        // Displayed fields
        public BehaviourType behavior = BehaviourType.Inference;
        [SerializeField] public PPOActor actor;
        [SerializeField] public NeuralNetwork critic;
        [HideInInspector] public PPOMemory memory;

        [Space,Min(1), SerializeField] private int observationSize = 2;
        [SerializeField] private ActionType actionSpace = ActionType.Continuous;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [Space, Min(10), Tooltip("seconds")] public int timeHorizon = 60;
        [SerializeField] private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;


        // Hidden fields
        [HideInInspector] public PPOHyperParameters hp;
        private TransformReseter personalEnvironment;
        RunningNormalizer observationsNormalizer;

        private AgentSensor agentSensor;
        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;

        private int episode = 1;
        private float episodeTimePassed = 0f;
        private double reward = 0; 
        private double cumulativeReward = 0;

        // Setup
        protected virtual void Awake()
        {
            hp = GetComponent<PPOHyperParameters>();

            InitNetwork();
           
            // Init memory
            memory = new PPOMemory(false);
           
            // Init buffers
            sensorBuffer = new SensorBuffer(observationSize);
            actionBuffer = new ActionBuffer(actor.GetNoParallelActions());
            
            // Init sensors
            agentSensor = new AgentSensor(this.transform);

            // Init environment
            personalEnvironment = onEpisodeEnd == OnEpisodeEndType.ResetEnvironment?
                                  personalEnvironment = new TransformReseter(this.transform.parent) :
                                  personalEnvironment = new TransformReseter(this.transform);

            // Subscribe to trainer instance
            if(behavior == BehaviourType.Inference)
                PPOTrainer.Subscribe(this);
        }
        private void InitNetwork()
        {
            // change the inspector vars if actor network exists
            if (actor)
            {
                observationSize = actor.GetNoObservations();
                
                if (actor.actionSpace == ActionType.Continuous)
                {
                    actionSpace = ActionType.Continuous;
                    ContinuousSize = actor.GetNoParallelActions();
                }
                else
                {
                    actionSpace = ActionType.Discrete;
                    DiscreteBranches = actor.outputBranches;
                }
                return;
            }

            // if doesn't exist, create the actor network
            if (actionSpace == ActionType.Discrete)
            {
                for (int i = 0; i < DiscreteBranches.Length; i++)
                    if (DiscreteBranches[i] < 1)
                    {
                        UnityEngine.Debug.LogError("Branch " + DiscreteBranches[i] + " cannot have 0 discrete actions");
                        throw new Exception("Error");
                    }
                actor = new PPOActor(observationSize, DiscreteBranches, hp.hiddenUnits, hp.layersNumber, hp.activationType, hp.initializationType);
            }
            else
            {
                if (ContinuousSize < 1)
                {
                    UnityEngine.Debug.LogError("Agent cannot have 0 continuous actions");
                    throw new Exception("Error");
                }
                actor = new PPOActor(observationSize, ContinuousSize, hp.hiddenUnits, hp.layersNumber, hp.activationType, hp.initializationType);              
            }

            // create the critic network
            if(critic == null && behavior == BehaviourType.Inference)
            {
                critic = new NeuralNetwork(observationSize, 1, hp.hiddenUnits, hp.layersNumber, hp.activationType, ActivationType.Linear, LossType.MeanSquare, InitializationType.NormalDistribution, true, GetCriticName());
                observationsNormalizer = new RunningNormalizer(observationSize);
            }
        }


        // Loop
        protected virtual void FixedUpdate()
        {
            switch (behavior)
            {
                case BehaviourType.Active:
                    SelfAction();
                    break;
                case BehaviourType.Inference:
                    InteractAction(false);
                    break;
                case BehaviourType.Manual:
                    ManualAction();
                    break;
            }
        }
        protected virtual void Update()
        {
            episodeTimePassed += Time.deltaTime;
            if (episodeTimePassed >= timeHorizon && behavior == BehaviourType.Inference)
                EndEpisode();
            
            if (behavior == BehaviourType.Inference && memory.IsFull(hp.buffer_size))
            {
                PPOTrainer.Ready();
            }
        }
        private void ManualAction()
        {
            actionBuffer.Clear();
            Heuristic(actionBuffer);
            OnActionReceived(actionBuffer);
        }
        private void SelfAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectObservations(sensorBuffer);
            agentSensor.CollectObservations(sensorBuffer);
            
            if(hp.normObservations) 
                observationsNormalizer.NormalizeMinusOneOne(sensorBuffer.Observations, false);

            if(actionSpace == ActionType.Continuous)
            {
                actionBuffer.ContinuousActions = actor.Forward_Continuous(sensorBuffer.Observations).Item2;           
            }
            else
            {
                actionBuffer.DiscreteActions = actor.Forward_Discrete(sensorBuffer.Observations).Item2;
            }
            OnActionReceived(actionBuffer);

        }
        private void InteractAction(bool isEpisodeEnd)
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectObservations(sensorBuffer);
            agentSensor.CollectObservations(sensorBuffer);
            
            if(hp.normObservations)
                observationsNormalizer.NormalizeMinusOneOne(sensorBuffer.Observations, true);

            if(actionSpace == ActionType.Continuous)
            {
                (double[], float[]) outs_acts = actor.Forward_Continuous(sensorBuffer.Observations);
                actionBuffer.ContinuousActions = outs_acts.Item2;

                double[] state = sensorBuffer.Observations;
                double[] action = outs_acts.Item1;
                double[] log_probs = PPOActor.GetContinuousLogProbs(outs_acts.Item1, outs_acts.Item2);
                double value = critic.Forward(sensorBuffer.Observations)[0];
                
                memory.Store(state, action, reward, log_probs, value, isEpisodeEnd);
            }
            else
            {
                (double[],int[]) dist_acts = actor.Forward_Discrete(sensorBuffer.Observations);
                actionBuffer.DiscreteActions = dist_acts.Item2;

                double[] state = sensorBuffer.Observations;
                double[] action = dist_acts.Item1;
                double[] log_probs = PPOActor.GetDiscreteLogProbs(dist_acts.Item1);
                double value = critic.Forward(sensorBuffer.Observations)[0];
                
                memory.Store(state, action, reward, log_probs, value, isEpisodeEnd);
            }
       
            reward = 0;
       
            OnActionReceived(actionBuffer);   
        }


        // User  
        public virtual void OnEpisodeBegin()
        {

        }
        public virtual void CollectObservations(SensorBuffer sensorBuffer)
        {

        }
        public virtual void OnActionReceived(in ActionBuffer actionBuffer)
        {

        }
        public virtual void Heuristic(ActionBuffer actionSet)
        {

        }
        public void AddReward<T>(T reward) where T : struct
        {         
            if (behavior == BehaviourType.Inactive) return;
            
            this.reward = Convert.ToDouble(reward);
            this.cumulativeReward += this.reward;
        }
        public void ForceAddReward<T>(T reward) where T : struct
        {
            this.reward = Convert.ToDouble(reward);
            this.cumulativeReward += this.reward;
        }
        public void EndEpisode()
        {
            // Collect last data piece (including the terminal reward)
            InteractAction(true);

            personalEnvironment?.Reset();
            OnEpisodeBegin();

            /*// Print statistics
            StringBuilder statistic = new StringBuilder();
            statistic.Append("<color=#0c74eb>");
            statistic.Append("Episode: ");
            statistic.Append(episode);
            statistic.Append(" | Time spent: ");
            statistic.Append(episodeTimePassed.ToString("0.0"));
            statistic.Append("s | Cumulated Reward: ");
            statistic.Append(cumulativeReward.ToString("0.000"));
            statistic.Append("</color>");
            UnityEngine.Debug.Log(statistic.ToString());*/

            episode++;
            episodeTimePassed = 0;
            cumulativeReward = 0;    
        }


        // Other
        private string GetCriticName()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<NeuralNetwork>("Assets/Critic#" + id + ".asset") != null)
                id++;
            return "Critic#" + id;
        }
        public ActionType GetActionSpace() => actionSpace;

    }












    [CustomEditor(typeof(PPOAgent), true), CanEditMultipleObjects]
    class ScriptlessAgent : Editor
    {
        public override void OnInspectorGUI()
        {
            var script = target as PPOAgent;
            List<string> dontDrawMe = new List<string>();
            dontDrawMe.Add("m_Script");

            // Hide action space
            SerializedProperty actType = serializedObject.FindProperty("actionSpace");
            if (actType.enumValueIndex == (int)ActionType.Continuous)
                dontDrawMe.Add("DiscreteBranches");
            else
                dontDrawMe.Add("ContinuousSize");

            // Hide networks
            SerializedProperty beh = serializedObject.FindProperty("behavior");
            if (beh.enumValueIndex == (int)BehaviourType.Manual)
            {
                dontDrawMe.Add("actor");
                dontDrawMe.Add("critic");
            }
            else if (beh.enumValueIndex == (int)BehaviourType.Active || 
                     beh.enumValueIndex == (int)BehaviourType.Inactive)
            {
                dontDrawMe.Add("critic");
            }
            

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}
