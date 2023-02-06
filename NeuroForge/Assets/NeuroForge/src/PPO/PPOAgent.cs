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
        public BehaviorType behavior = BehaviorType.Inference;
        [SerializeField] public PPONetwork model;
        [HideInInspector] public PPOMemory Memory;

        [Space,Min(1), SerializeField] private int observationSize = 2;
        [SerializeField] private ActionType actionSpace = ActionType.Continuous;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [Space, Min(10), Tooltip("seconds")] public int episodeMaxLength = 60;
        [SerializeField] private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;


        // Hidden fields
        [HideInInspector] public PPOHyperParameters hp;
        private TrainingEnvironment personalEnvironment;

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
            Memory = new PPOMemory(false);
           
            // Init buffers
            sensorBuffer = new SensorBuffer(observationSize);
            actionBuffer = new ActionBuffer(model.actorNetwork.GetActionsNumber());
            
            // Init sensors
            agentSensor = new AgentSensor(this.transform);

            // Init environment
            if (onEpisodeEnd == OnEpisodeEndType.ResetNone) return;
            personalEnvironment = onEpisodeEnd == OnEpisodeEndType.ResetEnvironment?
                                  personalEnvironment = new TrainingEnvironment(this.transform.parent) :
                                  personalEnvironment = new TrainingEnvironment(this.transform);

            // Subscribe to trainer instance
            if(behavior == BehaviorType.Inference)
                PPOTrainer.Subscribe(this);
        }
        private void InitNetwork()
        {
            if (model)
            {
                observationSize = model.actorNetwork.GetObservationsNumber();
                
                if (model.actorNetwork.actionSpace == ActionType.Continuous)
                {
                    actionSpace = ActionType.Continuous;
                    ContinuousSize = model.actorNetwork.GetActionsNumber();
                }
                else
                {
                    actionSpace = ActionType.Discrete;
                    DiscreteBranches = model.actorNetwork.outputBranches;
                }
                return;
            }

            // Create new model

            if (actionSpace == ActionType.Discrete)
            {
                for (int i = 0; i < DiscreteBranches.Length; i++)
                    if (DiscreteBranches[i] < 1)
                    {
                        UnityEngine.Debug.LogError("Branch " + DiscreteBranches[i] + " cannot have 0 discrete actions");
                        throw new Exception("Error");
                    }
                model = new PPONetwork(observationSize, DiscreteBranches, hp.hiddenUnits, hp.layersNumber, hp.activationType, hp.initializationType);
            }
            else
            {
                if (ContinuousSize < 1)
                {
                    UnityEngine.Debug.LogError("Agent cannot have 0 continuous actions");
                    throw new Exception("Error");
                }
                model = new PPONetwork(observationSize, ContinuousSize, hp.hiddenUnits, hp.layersNumber, hp.activationType, hp.initializationType);
            }
        }


        // Loop
        protected virtual void Update()
        {
            switch (behavior)
            {
                case BehaviorType.Self:
                    SelfAction();
                    break;
                case BehaviorType.Inference:
                    InteractAction(false);
                    break;
                case BehaviorType.Manual:
                    ManualAction();
                    break;
            }

            episodeTimePassed += Time.deltaTime;
            if (episodeTimePassed >= episodeMaxLength && behavior == BehaviorType.Inference)
                EndEpisode();
            
            if (behavior == BehaviorType.Inference && Memory.IsFull(hp.buffer_size))
            {
                PPOTrainer.Ready();
            }
        }
        private void ManualAction()
        {

        }
        private void SelfAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectObservations(sensorBuffer);
            agentSensor.CollectObservations(sensorBuffer);
            
            if(hp.normalizeObservations) 
                model.observationsNormalizer.NormalizeMinusOneOne(sensorBuffer.observations, false);

            if(actionSpace == ActionType.Continuous)
            {
                actionBuffer.continuousActions = model.actorNetwork.ContinuousForwardPropagation(sensorBuffer.observations).Item2;           
            }
            else
            {
                actionBuffer.discreteActions = model.actorNetwork.DiscreteForwardPropagation(sensorBuffer.observations).Item2;
            }
            OnActionReceived(actionBuffer);

        }
        private void InteractAction(bool isEpisodeEnd)
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectObservations(sensorBuffer);
            agentSensor.CollectObservations(sensorBuffer);
            
            if(hp.normalizeObservations)
                model.observationsNormalizer.NormalizeMinusOneOne(sensorBuffer.observations, true);

            if(actionSpace == ActionType.Continuous)
            {
                (double[], float[]) outs_acts = model.actorNetwork.ContinuousForwardPropagation(sensorBuffer.observations);
                actionBuffer.continuousActions = outs_acts.Item2;

                double[] state = sensorBuffer.observations;
                double[] action = outs_acts.Item1;
                double[] log_probs = model.actorNetwork.GetContinuousLogProbs(outs_acts.Item1, outs_acts.Item2);
                double value = model.criticNetwork.ForwardPropagation(sensorBuffer.observations)[0];
                
                Memory.Store(state, action, reward, log_probs, value, isEpisodeEnd);
            }
            else
            {
                (double[],int[]) dist_acts = model.actorNetwork.DiscreteForwardPropagation(sensorBuffer.observations);
                actionBuffer.discreteActions = dist_acts.Item2;

                double[] state = sensorBuffer.observations;
                double[] action = dist_acts.Item1;
                double[] log_probs = PPOActorNetwork.GetDiscreteLogProbs(dist_acts.Item1);
                double value = model.criticNetwork.ForwardPropagation(sensorBuffer.observations)[0];
                
                Memory.Store(state, action, reward, log_probs, value, isEpisodeEnd);
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
        public virtual void Heuristic(ActionBuffer actionBuffer)
        {

        }
        public void AddReward<T>(T reward) where T : struct
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
        public ActionType GetActionSpace() => actionSpace;

    }












    [CustomEditor(typeof(PPOAgent), true), CanEditMultipleObjects]
    class ScriptlessAgent : Editor
    {
        public override void OnInspectorGUI()
        {
            SerializedProperty actType = serializedObject.FindProperty("actionSpace");
            if (actType.enumValueIndex == (int)ActionType.Continuous)
            {
                DrawPropertiesExcluding(serializedObject, new string[] { "m_Script", "DiscreteBranches" });

            }
            else
            {
                DrawPropertiesExcluding(serializedObject, new string[] { "m_Script", "ContinuousSize" });
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
