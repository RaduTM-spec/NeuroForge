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
using System.Collections.ObjectModel;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        #region Public Fields
        public BehaviorType behavior = BehaviorType.Passive;

        [SerializeField] private ArtificialNeuralNetwork policyNetwork;
                         private ArtificialNeuralNetwork oldPolicyNetwork;
        [SerializeField] private ArtificialNeuralNetwork criticNetwork;

        [Space, SerializeField] private ExperienceBuffer experienceBuffer;

        [Space,Min(1), SerializeField] private int SpaceSize = 2;
        [Min(1), SerializeField] private int ActionSize = 2;
        [SerializeField] private ActionType actionType = ActionType.Continuous;
        #endregion

        #region Private Fields
        private int Episode = 1;
        private int Step = 0;//do not modify
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
            if (policyNetwork != null)
            {
                SpaceSize = policyNetwork.GetInputsNumber();
                ActionSize = policyNetwork.GetOutputsNumber();
                if (policyNetwork.outputActivationType == ActivationType.SoftMax)
                    actionType = ActionType.Discrete;
                else
                    actionType= ActionType.Continuous;
            }
               
            ActivationType activation = hp.activationType;
            ActivationType outputActivation = hp.activationType;
            LossType lossFunc = hp.lossType;

            int actorOutputs = ActionSize;
            if (actionType == ActionType.Discrete)
            {
                actorOutputs = ActionSize;
                outputActivation = ActivationType.SoftMax;
                lossFunc = LossType.CrossEntropy;
                
            }
            else // Continuous
            {
                actorOutputs = ActionSize * 2;
                outputActivation = ActivationType.Tanh;
            }
            
            if(policyNetwork == null) policyNetwork = new ArtificialNeuralNetwork(SpaceSize, actorOutputs, hp.networkHiddenLayers, activation, outputActivation, lossFunc, true, GetPolicyName());
            if(oldPolicyNetwork == null) oldPolicyNetwork = new ArtificialNeuralNetwork(policyNetwork, false, GetOldPolicyName());
            
            if (criticNetwork == null) criticNetwork = new ArtificialNeuralNetwork(SpaceSize, 1, hp.networkHiddenLayers, activation, ActivationType.None, LossType.MeanSquare, true, GetCriticName());
            if (experienceBuffer == null) experienceBuffer = new ExperienceBuffer(GetMemoryName(), true);

            experienceBuffer.Clear();

            string GetPolicyName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/Policy#" + id + ".asset") != null)
                    id++;
                return "PolicyNN#" + id;
            }
            string GetOldPolicyName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/OldPolicyNN#" + id + ".asset") != null)
                    id++;
                return "OldPolicyNN#" + id;
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
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/BufferXP#" + id + ".asset") != null)
                    id++;
                return "MemoryXP#" + id;
            }
        }
        private void InitBuffers()
        {
            sensorBuffer = new SensorBuffer(policyNetwork.GetInputsNumber());
            actionBuffer = new ActionBuffer(policyNetwork.GetOutputsNumber());
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
                    LearnAction();
                    break;
                default:
                    break;
            }
            Step++;
            if(hp.maxStep != 0 && Step >= hp.maxStep && behavior == BehaviorType.Inference) 
                EndEpisode();
        }
        private void ActiveAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensors(sensorBuffer);
            CollectObservations(sensorBuffer);

            double[] outs = policyNetwork.ForwardPropagation(sensorBuffer.observations);

            //Output form: (mean, stddev) (mean, stddev) (mean, stddev) => 3 continuous actions

            //Probability to Action
            if (actionType == ActionType.Continuous)
                for (int i = 0; i < outs.Length; i += 2)
                {
                    actionBuffer.actions[i / 2] = Functions.RandomGaussian(outs[i], outs[i + 1]); //UnityEngine.Random.Range(-1.0f, 1.0f) * outs[i+1] + outs[i]; 
                }

            OnActionReceived(actionBuffer);

        }
        private void LearnAction()
        {
            Collect_Action_Store(false);

            if (experienceBuffer.IsFull(hp.buffer_size) == true)
            {
                
                
                GAE(); // Normalize Rewards + Calculate Advantages + Normalize Advantages (should work fine)

                for (int i = 0; i < hp.buffer_size / hp.batch_size; i++) //Foreach miniBatch
                {
                    List<Sample> miniBatch = experienceBuffer.records.GetRange(i, i + hp.batch_size);
                    
                    lock(policyNetwork)lock(oldPolicyNetwork)lock(criticNetwork)
                    {
                        UpdateActor(miniBatch);
                        UpdateCritic(miniBatch);
                    }
                                 
                }
                
                oldPolicyNetwork.SetParametersFrom(policyNetwork);

                experienceBuffer.Clear();
            }
            reward = 0;
        }

        #endregion

        #region PPO
        private void Collect_Action_Store(bool episodeDone)
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensors(sensorBuffer);
            CollectObservations(sensorBuffer);

            double[] outs = policyNetwork.ForwardPropagation(sensorBuffer.observations);
            experienceBuffer.Store(sensorBuffer.observations, outs, reward, episodeDone);

            //Output form: (mean, stddev) (mean, stddev) (mean, stddev) => 3 continuous actions

            //Probability to Action
            if(actionType == ActionType.Continuous)
                for (int i = 0; i < outs.Length; i+=2)
                {
                    actionBuffer.actions[i/2] = Functions.RandomGaussian(outs[i], outs[i + 1]); //UnityEngine.Random.Range(-1.0f, 1.0f) * outs[i+1] + outs[i]; 
                }

            OnActionReceived(actionBuffer);  
        }
        private void GAE()
        {
            List<Sample> data = experienceBuffer.records;

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
            double advantage = 0.0;          
            for (int i = data.Count - 1; i >= 0; i--)
            {
                double value = criticNetwork.ForwardPropagation(data[i].state.Concat(data[i].action).ToArray())[0];

                double nextValue;
                if (i != data.Count - 1)
                { nextValue = criticNetwork.ForwardPropagation(data[i + 1].state.Concat(data[i + 1].action).ToArray())[0]; }
                else
                { nextValue = 0; }

                double delta;
                if (data[i].done)
                {
                    delta = data[i].reward - value;
                }
                else
                {
                    delta = reward + hp.discountFactor * nextValue - value;
                }
                advantage = advantage * hp.discountFactor * hp.gaeFactor + delta;

                data[i].advantage = advantage;
            }


            //Normalize advantages
            double mean = data.Sum(x => x.advantage) / data.Count;
            double std = Math.Sqrt(data.Sum(x => Math.Pow(x.advantage - mean, 2)) / data.Count);
            for (int i = 0; i < data.Count; i++)
            {
                data[i].advantage = (data[i].advantage - mean) / (std + 0.00000001);
            }
        }
        private void UpdateActor(List<Sample> mini_batch)
        {
            //convert s,a,adv to tensors (double arrays)
            double[][] states = mini_batch.Select(x => x.state).ToArray();
            double[][] actions = mini_batch.Select(x => x.action).ToArray();
            double[] advantages = mini_batch.Select(x => x.advantage).ToArray();

            for (int t = 0; t < states.Length; t++)
            {
                double[] old_log_policy_probs = oldPolicyNetwork.GetLogGaussianProb(states[t]);
                double[] new_log_policy_probs = policyNetwork.GetLogGaussianProb(states[t]);

                //RATIO
                double ratio = 1;
                for (int p = 0; p < old_log_policy_probs.Length; p++)
                {
                    ratio += new_log_policy_probs[p] - old_log_policy_probs[p];
                }

                //CLIPPED RATIO
                double clipped_ratio = Math.Min
                                       (
                                             ratio * advantages[t],
                                             Math.Clamp(ratio, 1 - hp.clipFactor, 1 + hp.clipFactor) * advantages[t]
                                       );

                //ENTROPY
                double entropy = 0;
                for (int p = 0; p < old_log_policy_probs.Length; p++)
                {
                    double ex = Math.Exp(new_log_policy_probs[p]);
                    entropy += -ex * Math.Log(ex);
                }
               

                //SURROGATE
                double surrogate_loss = clipped_ratio + hp.entropyRegularization * entropy;

               

                //SGD -> no Adam :(
                policyNetwork.BackwardPropagation(states[t], surrogate_loss);
                policyNetwork.UpdateParameters(hp.learnRate, hp.momentum, hp.regularization);
            }


        }
        private void UpdateCritic(List<Sample> mini_batch)
        {
            double[][] states = mini_batch.Select(x => x.state).ToArray();
            double[][] actions = mini_batch.Select(x => x.action).ToArray();
            double[] rewards = mini_batch.Select(x => x.reward).ToArray();
            double[] discounted_sum_rewards = CalculateDiscountedRewards(rewards);

            for (int i = 0; i < states.Length; i++)
            {
                double predicted_value = criticNetwork.ForwardPropagation(states[i])[0];

                double[] expectedValue = new double[] { discounted_sum_rewards[i] };

                //SGD
                criticNetwork.BackwardPropagation(states[i], expectedValue);
                criticNetwork.UpdateParameters(hp.learnRate, hp.momentum, hp.regularization);
            }
            
        }
        
        double[] CalculateDiscountedRewards(double[] rewards)
        {
            double[] d_s_m = new double[rewards.Length];

            double sum = 0;
            for (int i = rewards.Length - 1; i >= 0; i--)
            {
                sum = sum * hp.discountFactor + rewards[i];
                d_s_m[i] = sum;
            }
            return d_s_m;
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