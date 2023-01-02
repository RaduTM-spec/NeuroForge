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
using System.Runtime.CompilerServices;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        #region Public Fields
        public BehaviorType behavior = BehaviorType.Passive;
        [SerializeField] private ArtificialNeuralNetwork actor;
        [SerializeField] private ArtificialNeuralNetwork critic;
        [SerializeField] private Memory memory;

        [Space,Min(1), SerializeField] private int SpaceSize = 2;
        [Min(1), SerializeField] private int ActionSize = 2;
        [SerializeField] private ActionType actionType = ActionType.Continuous;
        #endregion

        #region Private Fields
        private int Episode = 1;
        private int Step = 0;//do not modify it
        private double episodeCumulatedReward = 0;

        private HyperParameters hp;
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
            hp = GetComponent<HyperParameters>();
            InitNetworks_InitMemory();
            InitBuffers();
            InitSensors(this.transform);
            InitEnvironment(this.transform.parent);
        }
        private void InitNetworks_InitMemory()
        {
            if (actor != null)
            {
                SpaceSize = actor.GetInputsNumber();
                ActionSize = actor.GetOutputsNumber();
                if (actor.outputActivationType == ActivationType.SoftMax)
                    actionType = ActionType.Discrete;
                else
                    actionType= ActionType.Continuous;
            }
               
            ActivationType activation = hp.activationType;
            ActivationType outputActivation = hp.activationType;
            LossType loss = hp.lossType;
            if (actionType == ActionType.Discrete)
            {
                outputActivation = ActivationType.SoftMax;
                loss = LossType.CrossEntropy;
                
            }
            else if (actionType == ActionType.Continuous)
            {
                outputActivation = ActivationType.Tanh;
            }

            if(actor == null) actor = new ArtificialNeuralNetwork(SpaceSize, ActionSize, hp.networkHiddenLayers, activation, outputActivation, loss, GetActorName());
            if(critic == null) critic = new ArtificialNeuralNetwork(SpaceSize + ActionSize, 1, hp.networkHiddenLayers, ActivationType.Tanh, ActivationType.Tanh, LossType.MeanSquare, GetCriticName());
            if (memory == null) memory = new Memory(GetMemoryName());

            memory.Clear();

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
            sensorBuffer = new SensorBuffer(actor.GetInputsNumber());
            actionBuffer = new ActionBuffer(actor.GetOutputsNumber());
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
                default:
                    break;
            }
            Step++;
            if(hp.maxStep != 0 && Step >= hp.maxStep && 
                (behavior == BehaviorType.Inference || behavior == BehaviorType.Heuristic))
                EndEpisode();
        }
        private void ActiveAction()
        {
            if(actor == null)
            {
                Debug.LogError("<color=red>Actor network is missing. Agent cannot take any actions.</color>");
                return;
            }

            CollectSensors(sensorBuffer);
            CollectObservations(sensorBuffer);
            actionBuffer.actions = actor.ForwardPropagation(sensorBuffer.observations);
            OnActionReceived(actionBuffer);

            sensorBuffer.Clear();
            actionBuffer.Clear();

        }
        private void LearnAction()
        {
            Collect_Action_Store(false);

            if (memory.IsFull(hp.memory_size) == true)
            {
                GAE();

                for (int i = 0; i < hp.memory_size / hp.mini_batch_size; i++)
                {
                    List<Sample> miniBatch = memory.records.GetRange(i, i + hp.mini_batch_size);
                    UpdateActor(miniBatch);
                    UpdateCritic(miniBatch);
                }
                memory.Clear();
            }
            reward = 0;
        }
        private void Collect_Action_Store(bool isEndOfEpisode)
        {
            
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensors(sensorBuffer);
            CollectObservations(sensorBuffer);
            if (behavior == BehaviorType.Inference)
            {
                actionBuffer.actions = actor.ForwardPropagation(sensorBuffer.observations);
                AddNoiseToActions(actionBuffer);
            }
            else if (behavior == BehaviorType.Heuristic)
            {
                Heuristic(actionBuffer);
            }
            OnActionReceived(actionBuffer);

            memory.Store(sensorBuffer.observations, actionBuffer.actions, reward, isEndOfEpisode);
        }
        #endregion

        #region PPO
        private void GAE()
        {
            List<Sample> data = memory.records;

            //Normalize rewards
            double minReward = data.Min(s => s.reward);
            double maxReward = data.Max(s => s.reward);
            for(int i = 0; i < data.Count; i++)
            {
                if (data[i].reward == 0)
                    continue;
                if (data[i].reward < 0)
                    data[i].reward /= -minReward;
                else
                    data[i].reward /= maxReward;
            }

            //Calculate advantages
            for (int i = data.Count - 1; i >= 0; i--)
            {
                double value = critic.ForwardPropagation(data[i].state.Concat(data[i].action).ToArray())[0];
                double nextValue;
                if (i != data.Count - 1)
                    nextValue = critic.ForwardPropagation(data[i + 1].state.Concat(data[i + 1].action).ToArray())[0];
                else
                    nextValue = 0;

                double reward = data[i].reward;
                double advantage = 0;
                if (data[i].done)
                {
                    advantage = reward - value;
                }
                else
                {
                    advantage = reward + hp.discountFactor * nextValue - value;
                }
                advantage *= hp.discountFactor * hp.gaeFactor;
                data[i].advantage = advantage;
            }


            //Normalize advantages
            double mean = data.Sum(x => x.advantage) / data.Count;
            double std = Math.Sqrt(data.Sum(x => Math.Pow(x.advantage - mean, 2)) / data.Count);

            for (int i = 0; i < data.Count; i++)
            {
                data[i].advantage = (data[i].advantage - mean) / std;
            }

        }
        private void UpdateCritic(List<Sample> mini_batch)
        {
            double[][] states = mini_batch.Select(x => x.state).ToArray();
            double[][] actions = mini_batch.Select(x => x.action).ToArray();
            double[] rewards = mini_batch.Select(x => x.reward).ToArray();
            double[] discounted_sum_rewards = GetDiscountedRewards();

            for (int i = 0; i < states.Length; i++)
            {
                double predicted_value = critic.ForwardPropagation(states[i].Concat(actions[i]).ToArray())[0];

                double[] expectedValue = new double[] { discounted_sum_rewards[i] };

                critic.BackPropagation(states[i].Concat(actions[i]).ToArray(), expectedValue);
            }
            critic.UpdateParameters(hp.learnRate, hp.momentum, hp.regularization);


            double[] GetDiscountedRewards()
            {
                double[] discounted_sum_rewards = new double[rewards.Length];

                double sum = 0;
                for (int i = rewards.Length - 1; i >= 0; i--)
                {
                    sum = sum * hp.discountFactor + rewards[i];
                    discounted_sum_rewards[i] = sum;
                }
                return discounted_sum_rewards;
            }
        }
        private void UpdateActor(List<Sample> mini_batch)
        {
            //convert s,a,adv to tensors (double arrays)

            double[][] states = mini_batch.Select(x => x.state).ToArray();    //this is a Tensor
            double[][] actions = mini_batch.Select(x => x.action).ToArray();
            double[] advantages = mini_batch.Select(x => x.advantage).ToArray();

            double[] old_policy_log_probs = actor.GetLogProbs(states, actions);

            //Gradient ascent
            for (int i = 0; i < states.Length; i++)
            {
                //Calculate new policy log probabilities
                double[] new_policy_log_probs = actor.GetLogProbs(new double[][] { states[i] }, new double[][] { actions[i] });

                //Policies ratio
                double r = Math.Exp(new_policy_log_probs[0] - old_policy_log_probs[i]);

                //Calculate surrogate loss
                double loss = -Math.Min(r * advantages[i], Math.Min(r,1 + hp.clipFactor) 
                                * 
                               Math.Max(r, 1- hp.clipFactor) * advantages[i]);

                //Generate label
                double[] expectedActions = new double[actions[i].Length];
                for (int j = 0; j < expectedActions.Length; j++)
                {
                    if (actions[i][j] == 1)
                        expectedActions[j] = loss;
                    else
                        expectedActions[j] = 0;
                }

                //Update the actor
                actor.BackPropagation(states[i], expectedActions);
            }
            actor.UpdateParameters(hp.learnRate, hp.momentum, hp.regularization);
        }
        

        #endregion

        #region Utils
        private void CollectSensors(SensorBuffer buffer)
        {
            foreach (var raySensor in raySensors)
            {
                buffer.AddObservation(raySensor.observations);
            }
            foreach (var camSensor in cameraSensors)
            {
                buffer.AddObservation(camSensor.FlatCapture());
            }
        }
        public void AddReward<T>(T reward) where T : struct
        {
            this.reward += Convert.ToDouble(reward);
            this.episodeCumulatedReward += Convert.ToDouble(reward);
        }
        public void AddStepPenalty<T>(T penalty) where T : struct
        {
            double t = hp.maxStep == 0 ? 1 : hp.maxStep;
            double ActionPenalty = -Math.Abs(Convert.ToDouble(penalty)) / t;
            AddReward(ActionPenalty);
        }
        public void EndEpisode()
        {
           
            Collect_Action_Store(true);

            int transformsStart = 0;
            ResetEnvironment(this.transform.parent, ref transformsStart);

            PrintEpsiodeStatistic();

            Episode++;
            Step = 0;
            reward = 0;
            episodeCumulatedReward = 0;


            void PrintEpsiodeStatistic()
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
        private void AddNoiseToActions(ActionBuffer actionBuffer)
        {
            if (hp.actionNoise == 0)
                return;
            for (int i = 0; i < actionBuffer.actions.Length; i++)
            {
                actionBuffer.actions[i] += Functions.RandomGaussian(0, hp.actionNoise);
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