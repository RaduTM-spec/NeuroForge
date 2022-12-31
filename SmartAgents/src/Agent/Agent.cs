using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using System.Reflection.Emit;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor.ProjectWindowCallback;
using System.Text;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        #region Public Fields
        public BehaviorType behavior = BehaviorType.Passive;
        [SerializeField] private ArtificialNeuralNetwork actorNetwork;
        [SerializeField] private ArtificialNeuralNetwork criticNetwork;
        [SerializeField] private Memory memory;

        [Space,Min(1), SerializeField] private int SpaceSize = 2;
        [Min(1), SerializeField] private int ActionSize = 2;
        [SerializeField] private ActionType actionType = ActionType.Continuous;

        [Space, SerializeField] private bool UseRaySensors = true;
        #endregion

        #region Private Fields
        private int Episode = 1;
        private int Step = 0;
        private double episodeCumulatedReward = 0;

        private HyperParameters hyperParameters;
        private List<RaySensor> raySensors = new List<RaySensor>();

        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;
        private double reward = 0;

        Sample lastFrameData = new Sample();
        List<Transform> initialEnvironmentState = new List<Transform>();
        #endregion

        #region Setup
        protected virtual void Awake()
        {
            hyperParameters = GetComponent<HyperParameters>();
            InitNetworks();
            InitBuffers();
            InitRaySensors(this.transform);
            InitEnvironment(this.transform.parent);
        }
        private void InitNetworks()
        {
            if (actorNetwork != null)
                return;

            ActivationType activation = hyperParameters.activationType;
            ActivationType outputActivation = hyperParameters.activationType;
            LossType loss = hyperParameters.lossType;
            
            if (actionType == ActionType.Discrete)
            {
                outputActivation = ActivationType.SoftMax;
                loss = LossType.CrossEntropy;
                
            }
            else if (actionType == ActionType.Continuous)
            {
                outputActivation = ActivationType.Tanh;
            }
            actorNetwork = new ArtificialNeuralNetwork(SpaceSize, ActionSize, hyperParameters.networkHiddenLayers, activation, outputActivation, loss, true, "ActorNN");
            criticNetwork = new ArtificialNeuralNetwork(SpaceSize + ActionSize, 1, hyperParameters.networkHiddenLayers, ActivationType.Tanh, ActivationType.Tanh, LossType.MeanSquare, true, "CriticNN");
            memory = new Memory("RecordXP");
        }
        private void InitBuffers()
        {
            sensorBuffer = new SensorBuffer(actorNetwork.GetInputsNumber());
            actionBuffer = new ActionBuffer(actorNetwork.GetOutputsNumber());
        }
        private void InitRaySensors(Transform parent)
        {
            //adds all 
            if (!UseRaySensors)
                return;

            RaySensor sensorFound = GetComponent<RaySensor>();
            if(sensorFound != null && sensorFound.enabled)
                raySensors.Add(sensorFound);
            foreach (Transform child in parent)
            {
                InitRaySensors(child);
            }
        }
        private void InitEnvironment(Transform parent)
        {
            foreach(Transform child in parent)
            {
                Transform clone = new GameObject().transform;

                clone.position = child.position;
                clone.rotation = child.rotation;
                clone.localScale = child.localScale;

                initialEnvironmentState.Add(clone);
                InitEnvironment(child);
            }
        }
        #endregion

        #region Loop
        protected virtual void Update()
        {
            
            switch(behavior)
            {
                case BehaviorType.Active:
                    ActiveAction();
                    break;
                case BehaviorType.Inference:
                case BehaviorType.Heuristic:
                    LearnAction();
                    break;
                default:
                    break;

            }

            Step++;
            if(hyperParameters.maxStep != 0 && Step > hyperParameters.maxStep)
            {
                //Add terminal reward
                if(episodeCumulatedReward <= 0)
                    AddReward(-1); 
                
                EndEpisode();
            }
        }
        private void ActiveAction()
        {
            if(actorNetwork == null)
            {
                Debug.LogError("<color=red>Actor network is missing. Agent cannot take any actions.</color>");
                return;
            }

            CollectRaySensorObservations();
            
            CollectObservations(sensorBuffer);
            actionBuffer.actions = actorNetwork.ForwardPropagation(sensorBuffer.observations);
            OnActionReceived(actionBuffer);

            sensorBuffer.Clear();
            actionBuffer.Clear();

        }
        private void LearnAction()
        {
            #region Collect Observation
            CollectRaySensorObservations();
            CollectObservations(sensorBuffer);
            #endregion

            #region Take Action

            if (behavior == BehaviorType.Inference) //Get predicted outs
            {
                actionBuffer = new ActionBuffer(actorNetwork.ForwardPropagation(sensorBuffer.observations));
                OnActionReceived(actionBuffer);
            }
            else // Get user outs
            {
                Heuristic(actionBuffer);
                OnActionReceived(actionBuffer);
            }

            #endregion

            #region Memory

            //Complete Previous
            lastFrameData.nextState = sensorBuffer.observations;
            lastFrameData.nextAction = actionBuffer.actions;

            //Add to memory
            if (memory && hyperParameters.memoryCapacity != 0 && lastFrameData.IsComplete())
            {
                if (memory.GetSize() >= hyperParameters.memoryCapacity)
                    memory.PopOldRecord();

                memory.AddRecord(lastFrameData);
            }

            //Init Current 
            lastFrameData.state = sensorBuffer.observations;
            lastFrameData.action = actionBuffer.actions;
            lastFrameData.reward = reward;

            #endregion

            if (!lastFrameData.IsComplete())
                return;

            #region Train Critic
           
            double[] criticInpFromLastFrame = lastFrameData.state.Concat(lastFrameData.action).ToArray();
            double[] criticInpFromCurrFrame = sensorBuffer.observations.Concat(actionBuffer.actions).ToArray();

            double tdTarget = reward + hyperParameters.discountFactor * criticNetwork.ForwardPropagation(criticInpFromCurrFrame)[0];
            double tdError = tdTarget - criticNetwork.ForwardPropagation(criticInpFromLastFrame)[0];

            criticNetwork.BackPropagation(criticInpFromLastFrame, new double[] { tdError }, true, hyperParameters.learnRate, hyperParameters.momentum, hyperParameters.regularization);
            
            #endregion

            #region Train Actor

            double AdvantageEstimate = tdTarget - actorNetwork.ForwardPropagation(lastFrameData.state).Average();
            actorNetwork.BackPropagation(lastFrameData.state, lastFrameData.action, true, hyperParameters.learnRate, hyperParameters.momentum, hyperParameters.regularization, AdvantageEstimate);

            #endregion


            sensorBuffer.Clear();
            actionBuffer.Clear();
            reward = 0;
        }
        private void CollectRaySensorObservations()
        {
            if (!UseRaySensors)
                return;

            foreach (var raySensor in raySensors)
            {
                sensorBuffer.AddObservation(raySensor.observations);
            }
        }
        #endregion

        #region Virtual
        public virtual void CollectObservations(SensorBuffer sensorBuffer)
        {

        }
        public virtual void OnActionReceived(ActionBuffer actionBuffer)
        {

        }
        public virtual void Heuristic(ActionBuffer actionBuffer)
        {

        }
        #endregion

        public void AddReward<T>(T reward) where T : struct
        {
            this.reward += Convert.ToDouble(reward);
            this.episodeCumulatedReward += Convert.ToDouble(reward);
        }
        public void AddActionPenalty<T>(T penalty) where T : struct
        {
            double t = hyperParameters.maxStep == 0 ? 1 : hyperParameters.maxStep;
            double ActionPenalty = -Math.Abs(Convert.ToDouble(penalty)) / t;
            AddReward(ActionPenalty);
        }
        public void EndEpisode()
        {
            //DO Graph statistics for Cumulative reward foreach episode and EpisodeLength foreach env using localhost:port 
            //Optimize training:
            // In on action received (virtualized)
             // AddReward(-1f/MaxStep) -> penalty agent foreach action it takes (only for optimizations)


            int start = 0;
            ResetEnvironment(this.transform.parent, ref start);

            StringBuilder statistic = new StringBuilder();
            statistic.Append("Episode: ");
            statistic.Append(Episode);
            statistic.Append(" | Steps: ");
            statistic.Append(Step);
            statistic.Append(" | Cumulated Reward: ");
            statistic.Append(episodeCumulatedReward);
            Debug.Log(statistic.ToString());

            Episode++;
            Step = 0;
            episodeCumulatedReward = 0;
        }
        private void ResetEnvironment(Transform parent, ref int index)
        {
            for (int i = 0; i < parent.transform.childCount; i++)
            {
                Transform child = parent.transform.GetChild(i);
                Transform initialTransform = initialEnvironmentState[index++];

                child.position = initialTransform.position;
                child.rotation = initialTransform.rotation;
                child.localScale = initialTransform.localScale;

                ResetEnvironment(child, ref index);
            }
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