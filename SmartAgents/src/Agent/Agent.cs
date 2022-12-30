using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using System.Reflection.Emit;
using System.Linq;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        #region Public Fields
        public BehaviorType behavior = BehaviorType.Passive;
        [SerializeField] private ArtificialNeuralNetwork actorNetwork;
        [SerializeField] private ArtificialNeuralNetwork criticNetwork;
        [SerializeField] private ExperienceRecord experienceRecord;

        [Space,Min(1), SerializeField] private int SpaceSize = 2;
        [Min(1), SerializeField] private int ActionSize = 2;
        [SerializeField] private ActionType actionType = ActionType.Continuous;

        [Space, SerializeField] private bool UseRaySensors = true;
        #endregion

        #region Private Fields
        private HyperParameters hyperParamters;
        private List<RaySensor> raySensors = new List<RaySensor>();

        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;
        private double reward = 0;

        private double totalReward = 0;
        #endregion

        #region Setup
        protected virtual void Awake()
        {
            hyperParamters = GetComponent<HyperParameters>();
            InitNetworks();
            InitBuffers();
            InitRaySensors(this.gameObject);  
        }
        private void InitNetworks()
        {
            if (actorNetwork != null)
                return;

            ActivationType activation = hyperParamters.activationType;
            ActivationType outputActivation = hyperParamters.activationType;
            LossType loss = hyperParamters.lossType;
            
            if (actionType == ActionType.Discrete)
            {
                outputActivation = ActivationType.SoftMax;
                loss = LossType.CrossEntropy;
                
            }
            else if (actionType == ActionType.Continuous)
            {
                outputActivation = ActivationType.Tanh;
            }
            actorNetwork = new ArtificialNeuralNetwork(SpaceSize, ActionSize, hyperParamters.networkHiddenLayers, activation, outputActivation, loss, true, "ActorNN");
            criticNetwork = new ArtificialNeuralNetwork(SpaceSize + ActionSize, 1, hyperParamters.networkHiddenLayers, ActivationType.Tanh, ActivationType.Tanh, LossType.MeanSquare, true, "CriticNN");
            experienceRecord = new ExperienceRecord("XPRecord");
        }
        private void InitBuffers()
        {
            sensorBuffer = new SensorBuffer(actorNetwork.GetInputsNumber());
            actionBuffer = new ActionBuffer(actorNetwork.GetOutputsNumber());
        }
        private void InitRaySensors(GameObject parent)
        {
            //adds all 
            if (!UseRaySensors)
                return;

            RaySensor sensorFound = GetComponent<RaySensor>();
            if(sensorFound.enabled)
                raySensors.Add(sensorFound);
            foreach (Transform child in parent.transform)
            {
                InitRaySensors(child.gameObject);
            }
        }
        #endregion

        #region Loop
        protected virtual void Update()
        {
            sensorBuffer.Clear();
            sensorBuffer.Clear();
            switch(behavior)
            {
                case BehaviorType.Active:
                    ActiveAction();
                    break;
                case BehaviorType.Manual:
                    ManualAction();
                    break;
                case BehaviorType.Learn:
                    LearnAction();
                    break;
                case BehaviorType.Heuristic:
                    HeuristicAction();
                    break;
                default:
                    break;

            }
        }
        private void ActiveAction()
        {
            if(actorNetwork == null)
            {
                Debug.LogError("<color=red>Compound network is missing. Agent cannot take any actions.</color>");
                return;
            }

            if(UseRaySensors)
            {
                foreach (var raySensor in raySensors)
                {
                    sensorBuffer.AddObservation(raySensor);
                }
            }
            
            CollectObservations(sensorBuffer);
            actionBuffer.actions = actorNetwork.ForwardPropagation(sensorBuffer.observations);
            OnActionReceived(actionBuffer);

        }
        private void ManualAction()
        {
            Heuristic(actionBuffer);
            OnActionReceived(actionBuffer);
        }
        private void LearnAction()
        {
            //EVENTUALLY ADD OFFLINE LEARN(from ExperienceRecord)
            //Collect observations and Take action
            CollectObservations(sensorBuffer);//agent collects observations
            actionBuffer = new ActionBuffer(actorNetwork.ForwardPropagation(sensorBuffer.observations));//network predicts actions
            OnActionReceived(actionBuffer); // agent takes action based on previous prediction

            //Update critic network 
            double[] criticInputs = sensorBuffer.observations.Concat(actionBuffer.actions).ToArray();
            double tdTarget = reward +
                              hyperParamters.discountFactor *
                              criticNetwork.ForwardPropagation(criticInputs)[0];

            double tdError = tdTarget - criticNetwork.ForwardPropagation(criticInputs)[0];

            criticNetwork.BackPropagation(criticInputs, new double[] { tdError }, true, hyperParamters.learnRate, hyperParamters.momentum, hyperParamters.regularization);

            //Update actor network
        }
        private void HeuristicAction()
        {
            CollectObservations(sensorBuffer);
            Heuristic(actionBuffer);

            double[] predictions = actorNetwork.ForwardPropagation(sensorBuffer.observations);
            ActionBuffer predictionsBuffer = new ActionBuffer(predictions.Length);
            predictionsBuffer.actions = predictions;

            double error = actorNetwork.BackPropagation(sensorBuffer.observations, actionBuffer.actions, true, hyperParamters.learnRate, hyperParamters.momentum, hyperParamters.regularization);
            Debug.Log("Error: " + error);
        }
        #endregion

        #region Virtual
        public virtual void CollectObservations(SensorBuffer sensorBuffer)
        {

        }
        public virtual void OnActionReceived(ActionBuffer actionBuffer)
        {

        }
        public virtual void Heuristic(ActionBuffer actionSet)
        {

        }
        #endregion

        public void AddReward<T>(T reward) where T : struct
        {
            this.reward += Convert.ToDouble(reward);
        }
        public void EndAction()
        {
          
        }

        
    }
    #region Custom Editor
    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    class ScriptlessAgent : Editor
    {
        private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawPropertiesExcluding(serializedObject, _dontIncludeMe);

            serializedObject.ApplyModifiedProperties();
        }
    }
    #endregion
}