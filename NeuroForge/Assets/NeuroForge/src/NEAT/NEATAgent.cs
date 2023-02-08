using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [DisallowMultipleComponent, RequireComponent(typeof(NEATHyperParameters))]
    public class NEATAgent : MonoBehaviour
    {
        #region Fields
        public BehaviourType behaviour = BehaviourType.Inference;
        [SerializeField] public NEATNetwork model;

        [Space]
        [Min(1), SerializeField] private int observationSize = 2;
        [SerializeField] private ActionType actionSpace = ActionType.Continuous;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [HideInInspector] public NEATHyperParameters hp;

        private AgentSensor agentSensor;
        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;

        private int speciesNumber;
        private float fitness;

        #endregion

        // Setup
        protected virtual void Awake()
        {
            hp = GetComponent<NEATHyperParameters>();

            InitNetwork();

            // Init buffers
            sensorBuffer = new SensorBuffer(model.GetInputsNumber());
            actionBuffer = new ActionBuffer(model.GetOutputsNumber());

            // Init sensors
            agentSensor = new AgentSensor(this.transform);

            // Init trainer
            if (behaviour == BehaviourType.Inference)
                NEATTrainer.Initialize(this);
        }
        public void InitNetwork()
        {
            if (model)
            {
                observationSize = model.GetInputsNumber();
                actionSpace = model.actionSpace;
                ContinuousSize = model.GetOutputsNumber();
                DiscreteBranches = model.outputShape;
                return;
            }


            int[] outputShape;
            if (actionSpace == ActionType.Continuous)
            {
                if (ContinuousSize < 1)
                {
                    UnityEngine.Debug.LogError("Agent cannot have 0 continuous actions!");
                    return;
                }
                outputShape = new int[1];
                outputShape[0] = ContinuousSize;
            }
            else
            {
                // Check if Discrete branches are correct
                for (int i = 0; i < DiscreteBranches.Length; i++)
                    if (DiscreteBranches[i] < 1)
                    {
                        Debug.LogError("Branch " + DiscreteBranches[i] + " cannot have 0 discrete actions!");
                        return;
                    }
                outputShape = DiscreteBranches;
            }

            model = new NEATNetwork(observationSize, outputShape, actionSpace, true);
        }


        // Loop
        protected virtual void Update()
        {
            switch (behaviour)
            {
                case BehaviourType.Active:
                    ActiveAction();
                    break;
                case BehaviourType.Inference:
                    ActiveAction();
                    break;
                case BehaviourType.Manual:
                    ManualAction();
                    break;
                default:
                    // Inactive
                    break;
            }
        }
        private void ManualAction()
        {

        }
        private void ActiveAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectObservations(sensorBuffer);
            agentSensor.CollectObservations(sensorBuffer);

            if (actionSpace == ActionType.Continuous)
            {
                actionBuffer.continuousActions = model.GetContinuousActions(sensorBuffer.observations);
            }
            else
            {
                actionBuffer.discreteActions = model.GetDiscreteActions(sensorBuffer.observations);
            }
            OnActionReceived(actionBuffer);
        }
       


        // User Call 
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
            if (behaviour == BehaviourType.Inactive) return;          
            this.fitness += Convert.ToSingle(reward);
        }
        public void EndEpisode()
        {
            if(behaviour == BehaviourType.Inference)
            {
                behaviour = BehaviourType.Inactive;
                NEATTrainer.Ready();
            }
            
        }


        // Other
        public float GetFitness() => fitness;
        public int GetSpecieNumber() => speciesNumber;
        public void SetSpecieNumber(int specieNumber) => speciesNumber = specieNumber;
        public void Resurrect()
        {
            behaviour = BehaviourType.Inference;
            fitness = 0f;
        }
        public ActionType GetActionSpace() => actionSpace;
    }




















    #region Custom Editor
    [CustomEditor(typeof(NeuroForge.NEATAgent), true), CanEditMultipleObjects]
    class ScriptlessNEATAgent : Editor
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
    #endregion
}

