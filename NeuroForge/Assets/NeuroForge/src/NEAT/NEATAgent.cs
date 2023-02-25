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
        public BehaviourType behavior = BehaviourType.Inference;
        [SerializeField] public Genome model;
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
        private float fitness = 0;
        #endregion

        // Setup
        protected virtual void Awake()
        {
            hp = GetComponent<NEATHyperParameters>();

            InitNetwork();

            // Init Innovation Counter
            InnovationHistory.Instance = new InnovationHistory(this.model);

            // Init buffers
            sensorBuffer = new SensorBuffer(model.GetInputsNumber());
            actionBuffer = new ActionBuffer(model.GetOutputsNumber());

            // Init sensors
            agentSensor = new AgentSensor(this.transform);

            // Init trainer
            if (behavior == BehaviourType.Inference)
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

            model = new Genome(observationSize, outputShape, actionSpace, fullyConnected, true);
        }


        // Loop
        protected virtual void Update()
        {
            switch (behavior)
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
            if (behavior == BehaviourType.Inactive) return;          
            this.fitness += Convert.ToSingle(reward);
        }
        public void ForceAddReward<T>(T reward) where T : struct
        {
            this.fitness += Convert.ToSingle(reward);
        }
        public void SetReward<T>(T reward)
        {
            this.fitness = Convert.ToSingle(reward);
        }
        public void EndEpisode()
        {
            if(behavior == BehaviourType.Inference)
            {
                behavior = BehaviourType.Inactive;
                NEATTrainer.Ready();
            }
            
        }
       


        // Other
        public float GetFitness() => fitness;
        public Species GetSpecies() => species;
        public void SetSpecies(Species species) => this.species = species;
        public void Resurrect() => behavior = BehaviourType.Inference;
        public void ResetFitness() => fitness = 0f;
        public ActionType GetActionSpace() => actionSpace;
    }




















    #region Custom Editor
    [CustomEditor(typeof(NeuroForge.NEATAgent), true), CanEditMultipleObjects]
    class ScriptlessNEATAgent : Editor
    {
        public override void OnInspectorGUI()
        {
            var script = target as NEATAgent;
            List<string> dontDrawMe = new List<string>();
            dontDrawMe.Add("m_Script");

            // Hide action space
            SerializedProperty actType = serializedObject.FindProperty("actionSpace");
            if (actType.enumValueIndex == (int)ActionType.Continuous)
                dontDrawMe.Add("DiscreteBranches");
            else
                dontDrawMe.Add("ContinuousSize");

            // Hide network and fully con
            SerializedProperty beh = serializedObject.FindProperty("behavior");

            if (beh.enumValueIndex == (int)BehaviourType.Manual)
            {
                dontDrawMe.Add("model");
                dontDrawMe.Add("fullyConnected");
            }
            else if (beh.enumValueIndex == (int)BehaviourType.Active ||
                     beh.enumValueIndex == (int)BehaviourType.Inactive)
            {
                dontDrawMe.Add("critic");
                dontDrawMe.Add("fullyConnected");
            }
            else if(script.model != null) // and inference behaviour
            {
                dontDrawMe.Add("fullyConnected");
            }

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
    #endregion
}

