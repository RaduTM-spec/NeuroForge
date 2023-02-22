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
        [SerializeField] private bool fullyConnected = false;

        [Space]
        [Min(1), SerializeField] private int observationSize = 2;
        [Min(1), SerializeField] private ActionType actionSpace = ActionType.Discrete;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [HideInInspector] public NEATHyperParameters hp;

        private AgentSensor agentSensor;
        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;

        private Species species;
        [SerializeField] private float fitness = 0;
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

            model = new NEATNetwork(observationSize, outputShape, actionSpace, fullyConnected, true);
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
        public void ForceAddReward<T>(T reward) where T : struct
        {
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
        public Species GetSpecies() => species;
        public void SetSpecies(Species species) => this.species = species;
        public void Resurrect() => behaviour = BehaviourType.Inference;
        public void ResetFitness() => fitness = 0f;
        public ActionType GetActionSpace() => actionSpace;
    }




















    #region Custom Editor
    [CustomEditor(typeof(NeuroForge.NEATAgent), true), CanEditMultipleObjects]
    class ScriptlessNEATAgent : Editor
    {
        public override void OnInspectorGUI()
        {
            SerializedProperty actType = serializedObject.FindProperty("actionSpace");
            List<string> what_not_to_draw = new List<string>();
            what_not_to_draw.Add("m_Script");

            if (actType.enumValueIndex == (int)ActionType.Continuous)
            {
                what_not_to_draw.Add("DiscreteBranches");
                
            }
            else
            {
                what_not_to_draw.Add("ContinuousSize");
            }
         
            var script = target as NEATAgent;
            if (script.model != null)
            {
                what_not_to_draw.Add("fullyConnected");
            }

            DrawPropertiesExcluding(serializedObject, what_not_to_draw.ToArray());

            serializedObject.ApplyModifiedProperties();
        }
    }
    #endregion
}

