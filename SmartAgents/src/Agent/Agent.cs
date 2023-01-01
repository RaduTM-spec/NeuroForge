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
using UnityEngine.Profiling;
using System.Diagnostics.CodeAnalysis;

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
        #endregion

        #region Private Fields
        private int Episode = 1;
        private int Step = 0;//do not modify it
        private double episodeCumulatedReward = 0;

        private HyperParameters hyperParameters;
        private List<RaySensor> raySensors = new List<RaySensor>();
        private List<CameraSensor> cameraSensors = new List<CameraSensor>();

        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;
        private double reward = 0;

        List<Transform> initialEnvironmentState = new List<Transform>();
        #endregion

        #region Setup
        protected virtual void Awake()
        {
            hyperParameters = GetComponent<HyperParameters>();
            InitNetworks_InitMemory();
            InitBuffers();
            InitSensors(this.transform);
            InitEnvironment(this.transform.parent);
        }
        private void InitNetworks_InitMemory()
        {
            if (actorNetwork != null)
            {
                SpaceSize = actorNetwork.GetInputsNumber();
                ActionSize = actorNetwork.GetOutputsNumber();
                if (actorNetwork.outputActivationType == ActivationType.SoftMax)
                    actionType = ActionType.Discrete;
                else
                    actionType= ActionType.Continuous;
            }
               

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

            if(actorNetwork == null) actorNetwork = new ArtificialNeuralNetwork(SpaceSize, ActionSize, hyperParameters.networkHiddenLayers, activation, outputActivation, loss, GetActorName());
            if(criticNetwork == null) criticNetwork = new ArtificialNeuralNetwork(SpaceSize + ActionSize, 1, hyperParameters.networkHiddenLayers, ActivationType.Tanh, ActivationType.Tanh, LossType.MeanSquare, GetCriticName());
            if(memory == null) memory = new Memory(GetMemoryName());

            string GetActorName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/ActorNN#" + id + ".asset") != null)
                    id++;
                return "ActorNN#" + id;
            }
            string GetCriticName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/CriticNN#" + id + ".asset") != null)
                    id++;
                return "CriticNN#" + id;
            }
            string GetMemoryName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/MemoryXP#" + id + ".asset") != null)
                    id++;
                return "MemoryXP#" + id;
            }
        }
        private void InitBuffers()
        {
            sensorBuffer = new SensorBuffer(actorNetwork.GetInputsNumber());
            actionBuffer = new ActionBuffer(actorNetwork.GetOutputsNumber());
        }
        private void InitSensors(Transform parent)
        {
            RaySensor rayFound = GetComponent<RaySensor>();
            CameraSensor camFound = GetComponent<CameraSensor>();
            if(rayFound != null && rayFound.enabled)
                raySensors.Add(rayFound);
            if(camFound != null && camFound.enabled)
                cameraSensors.Add(camFound);
            foreach (Transform child in parent)
            {
                InitSensors(child);
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


                case BehaviorType.OfflineCollectData:
                    CollectData(false);
                    break;
                case BehaviorType.OfflineTrainCritic:
                    TrainCritic();
                    break;
                case BehaviorType.OfflineTrainActor:
                    TrainActor();
                    break;
                default:
                    break;

            }

            Step++;
            if(hyperParameters.maxStep != 0 && Step >= hyperParameters.maxStep && 
                (behavior == BehaviorType.Inference || behavior == BehaviorType.Heuristic))
                EndEpisode();
        }
        private void ActiveAction()
        {
            if(actorNetwork == null)
            {
                Debug.LogError("<color=red>Actor network is missing. Agent cannot take any actions.</color>");
                return;
            }

            CollectSensors();
            
            CollectObservations(sensorBuffer);
            actionBuffer.actions = actorNetwork.ForwardPropagation(sensorBuffer.observations);
            OnActionReceived(actionBuffer);

            sensorBuffer.Clear();
            actionBuffer.Clear();

        }
        private void LearnAction()
        {
           /* #region Collect Observation
            CollectSensors();
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

            //Add to memory
            if (memory && hyperParameters.memoryCapacity != 0 && lastFrameData.IsComplete())
            {
                if (memory.GetSize() >= hyperParameters.memoryCapacity)
                    memory.PopFirst();

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

            criticNetwork.BackPropagation(criticInpFromLastFrame, new double[] { tdError });
            criticNetwork.UpdateParameters(hyperParameters.learnRate, hyperParameters.momentum, hyperParameters.regularization);
            #endregion

            #region Train Actor

            double AdvantageEstimate = tdTarget - actorNetwork.ForwardPropagation(lastFrameData.state).Average();
            actorNetwork.BackPropagation(lastFrameData.state, lastFrameData.action, AdvantageEstimate);
            actorNetwork.UpdateParameters(hyperParameters.learnRate, hyperParameters.momentum, hyperParameters.regularization);
            #endregion


            sensorBuffer.Clear();
            actionBuffer.Clear();
            reward = 0;*/
        }

        #endregion

        #region Utils
        private void CollectSensors()
        {
            foreach (var raySensor in raySensors)
            {
                sensorBuffer.AddObservation(raySensor.observations);
            }
            foreach (var camSensor in cameraSensors)
            {
                sensorBuffer.AddObservation(camSensor.FlatCapture());
            }
        }
        public void AddReward<T>(T reward) where T : struct
        {
            this.reward += Convert.ToDouble(reward);
            this.episodeCumulatedReward += Convert.ToDouble(reward);
        }
        public void AddStepPenalty<T>(T penalty) where T : struct
        {
            double t = hyperParameters.maxStep == 0 ? 1 : hyperParameters.maxStep;
            double ActionPenalty = -Math.Abs(Convert.ToDouble(penalty)) / t;
            AddReward(ActionPenalty);
        }
        public void EndEpisode()
        {
            if (behavior == BehaviorType.OfflineCollectData)
                CollectData(true);

            int transformsStart = 0;
            ResetEnvironment(this.transform.parent, ref transformsStart);

            PrintEpsiodeStatistic();

            Episode++;
            Step = 0;
            reward = 0;
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

        public virtual void CollectObservations(SensorBuffer sensorBuffer)
        {

        }
        public virtual void OnActionReceived(in ActionBuffer actionBuffer)
        {

        }
        public virtual void Heuristic(ActionBuffer actionBuffer)
        {

        }
        #endregion

        


        List<Sample> localData = new List<Sample>();
        private void CollectData(bool isEndOfEpisode)
        {
            //End
            if (localData.Count == hyperParameters.memoryCapacity)
            {
                localData.RemoveAt(localData.Count - 1);
                Debug.Log("<color=#00ff2f>Memory fulfilled </color><color=blue>(" + hyperParameters.memoryCapacity + ")</color>");

                #region Normalize Rewards
                    double minReward = localData.Min(x => x.reward);
                    double maxReward = localData.Max(x => x.reward);
                    
                    for (int i = 0; i < localData.Count; i++)
                    {
                        double normalizedReward;
                        if (localData[i].reward < 0)
                            normalizedReward = -(localData[i].reward / minReward);
                        else
                            normalizedReward = localData[i].reward / maxReward;
                        localData[i].reward = normalizedReward;
                    }
                #endregion

                memory.records = localData;
                memory.CalculateDiscountedRewards(hyperParameters.discountFactor, criticNetwork);

                EditorApplication.ExitPlaymode();
                return;
            }


            #region STATE ACTION
            CollectSensors();
            CollectObservations(sensorBuffer);
            Heuristic(actionBuffer);
            OnActionReceived(actionBuffer);
            #endregion

            //Update previous frame
            if (Step > 0)
            {
                Sample previousFrame = localData.Last<Sample>();
                previousFrame.nextState = sensorBuffer.observations;
                previousFrame.nextAction = actionBuffer.actions;
            }
            //Collect current frame
            localData.Add(new Sample(sensorBuffer.observations, actionBuffer.actions, reward, isEndOfEpisode));
            
            sensorBuffer.Clear();
            actionBuffer.Clear();
            reward = 0;

           
        }
        private void TrainCritic(float batchSplit = 0.1f)
        {
            // random MiniBatch
            //Each frame is an epoch

            double epochErr = 0;

            int sampleId = 0;
            int updateFlag = (int)(memory.records.Count * batchSplit);
            
            foreach (var sample in memory.records)
            {
                double[] inputs = sample.state.Concat(sample.action).ToArray();
                double error = criticNetwork.BackPropagation(inputs, new double[] { sample.reward });
                epochErr += error;
               
            }
            criticNetwork.UpdateParameters(hyperParameters.learnRate, hyperParameters.momentum, hyperParameters.regularization);

            
            double accuracy = (1 - epochErr / memory.records.Count) * 100;
            hyperParameters.progressChart.AddKey(Step, (float)accuracy);
            hyperParameters.epoch++;
            hyperParameters.accuracy = accuracy.ToString("0.00000") + "%";            
        }
        private void TrainActor()
        {
            //calculate expected returns using critic foreach sample
            // expected Return = sample.reward + gamma * critic.Predict(s')


            // advantage error = expectedReturn - critic.Predict(s,a)
            // actor.BackProp(s,a, advantage_error);



        }

        private void PrintEpsiodeStatistic()
        {
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Episode: ");
            statistic.Append(Episode);
            statistic.Append(" | Steps: ");
            statistic.Append(Step);
            statistic.Append(" | Cumulated Reward: ");
            statistic.Append(episodeCumulatedReward);
            Debug.Log(statistic.ToString());
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