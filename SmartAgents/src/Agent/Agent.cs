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
        [SerializeField] private ArtificialNeuralNetwork criticNetwork;

        [Space, SerializeField] private ExperienceBuffer experienceBuffer;

        [Space,Min(1), SerializeField] private int SpaceSize = 2;
        [Min(1), SerializeField] private int ActionSize = 2;
        [SerializeField] private ActionType actionType = ActionType.Continuous;
        #endregion

        #region Private Fields
        private int Episode = 1;
        private int Step = 0;
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
                outputActivation = ActivationType.Tanh_and_Softplus;
            }
            
            if(policyNetwork == null) policyNetwork = new ArtificialNeuralNetwork(SpaceSize, actorOutputs, hp.networkHiddenLayers, activation, outputActivation, lossFunc, true, GetPolicyName());
            //if(oldPolicyNetwork == null) oldPolicyNetwork = new ArtificialNeuralNetwork(policyNetwork, false, GetOldPolicyName());
            
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
                double[] returns = GAE();
                
                for (int i = 0; i < hp.buffer_size / hp.batch_size; i++) 
                {
                    List<Sample> miniBatch = experienceBuffer.records.GetRange(i, i + hp.batch_size);
                    
                    lock(policyNetwork)lock(criticNetwork)
                    {
                        UpdateActorCritic(miniBatch, returns);
                    }                                
                }     
                experienceBuffer.Clear();
            }
            reward = 0;
        }

        #endregion

        #region PPO Training
        private void Collect_Action_Store(bool episodeDone)
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensors(sensorBuffer);
            CollectObservations(sensorBuffer);

            double[] rawOutputs = policyNetwork.ForwardPropagation(sensorBuffer.observations);
            
            //Discrete output form: 1branch(x,y,z) x+y+z = 1 (softmax)
            //Continous output form: (mean, stddev) (mean, stddev) (mean, stddev) => 3 continuous actions

            double[] log_probs = GetLogProbs(rawOutputs);
            double value = criticNetwork.ForwardPropagation(sensorBuffer.observations)[0];

            experienceBuffer.Store(sensorBuffer.observations, rawOutputs, reward, log_probs, value, episodeDone);

            FillActionBuffer(rawOutputs);
            OnActionReceived(actionBuffer);  
        }
        private double[] GAE()
        {
            //Advantages are stored inside the buffer
            List<Sample> data = experienceBuffer.records;

            //Returns are returned
            List<double> returns = new List<double>();

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
            double advantage = 0.0;  //or gae value
            for (int i = data.Count - 1; i >= 0; i--)
            {
                double value = criticNetwork.ForwardPropagation(data[i].state)[0];
                double nextValue = i == data.Count-1? 
                                        0 : 
                                        criticNetwork.ForwardPropagation(data[i + 1].state)[0];


                double delta = data[i].done? 
                                data[i].reward - value :
                                reward + hp.discountFactor * nextValue - value; //Bellman equation

                advantage = advantage * hp.discountFactor * hp.gaeFactor + delta;

                returns.Add(advantage + value);
            }
            returns.Reverse();
            return returns.ToArray();
        }
        private void UpdateActorCritic(List<Sample> mini_batch, double[] returns)
        {
            //convert s,a,adv to tensors (double arrays)
            double[][] states = mini_batch.Select(x => x.state).ToArray();
            double[][] actions = mini_batch.Select(x => x.action).ToArray();

            //calculate and normalize advantages
            double[] advantages = mini_batch.Select(x => -x.value).ToArray();
            for (int i = 0; i < advantages.Length; i++)
            {
                advantages[i] += returns[i];
            }
            NormalizeAdvantages(advantages, mini_batch.Count);

            //Train
            for (int t = 0; t < mini_batch.Count; t++)
            {
                double[] rawOutput = policyNetwork.ForwardPropagation(mini_batch[t].state);
                double value = criticNetwork.ForwardPropagation(mini_batch[t].state)[0];

                double[] old_policy_log_probs = mini_batch[t].log_probs;
                double[] new_policy_log_probs = GetLogProbs(rawOutput);

                //RATIO
                double[] ratios = new double[old_policy_log_probs.Length];
                for (int p = 0; p < old_policy_log_probs.Length; p++)
                {
                    ratios[p] = new_policy_log_probs[p] - old_policy_log_probs[p];
                }

                double[] surrogate_losses = new double[ratios.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    surrogate_losses[r] = -Math.Min
                                       (
                                             ratios[r] * advantages[t],
                                             Math.Clamp(ratios[r], 1 - hp.clipFactor, 1 + hp.clipFactor) * advantages[t]
                                       );
                }

                //ENTROPY
                List<double> entrops = new List<double>();
                for (int o = 0; o < rawOutput.Length; o+=2)
                {
                    double mu = rawOutput[o];
                    double sigma = rawOutput[o + 1];
                    entrops.Add(0.5 * Math.Log(2 * Math.PI * Math.E * sigma * sigma));
                }
                double[] entropies = entrops.ToArray();

                //SURROGATE LOSS
                double[] critic_loss = new double[] { returns[t] - value };

                double[] actor_loss = new double[rawOutput.Length];
                for (int i = 0; i < actor_loss.Length; i++)
                {
                    actor_loss[i] = surrogate_losses[i/2] + hp.entropyRegularization * entropies[i/2];

                }

                //SGD
                criticNetwork.BackwardPropagation(states[t], critic_loss);
                criticNetwork.UpdateParameters(hp.learnRate, hp.momentum, hp.regularization);
                
                policyNetwork.BackwardPropagation(states[t], actor_loss);
                policyNetwork.UpdateParameters(hp.learnRate, hp.momentum, hp.regularization);
            }


        }
        
        void NormalizeAdvantages(double[] advantages, int dataCount)
        {
            //Normalize advantages
            double mean = advantages.Sum() / advantages.Length;
            double std = Math.Sqrt(advantages.Sum(x => Math.Pow(x - mean, 2) / advantages.Length));
            for (int i = 0; i < dataCount; i++)
            {
                advantages[i] = (advantages[i] - mean) / (std + 0.00000001);
            }
        }
        double[] GetLogProbs(double[] rawOutputs)
        {
            List<double> log_probs = new List<double>();
            if (actionType == ActionType.Discrete)
            {
                // to be completed
            }
            else if (actionType == ActionType.Continuous)
            {
                for (int i = 0; i < rawOutputs.Length; i += 2)
                {
                    double mean = rawOutputs[i];
                    double stddev = rawOutputs[i + 1];

                    double actionSample = Math.Clamp(Functions.RandomGaussian(mean, stddev),-1,1);

                    double log_prob = LogProb(actionSample, mean, stddev);
                    log_probs.Add(log_prob);

                }
            }
            return log_probs.ToArray();

            double LogProb(double action, double mu, double sigma)
            {
                double logProb = -0.5 * Math.Log(2 * Math.PI * sigma * sigma) - ((action - mu) * (action - mu)) / (2 * sigma * sigma);
                return logProb;
            }
        }
        void FillActionBuffer(double[] rawOutputs)
        {
            if (actionType == ActionType.Discrete)
            {
                actionBuffer.actions = rawOutputs;
            }
            else if (actionType == ActionType.Continuous)
            {
                for (int i = 0; i < rawOutputs.Length; i += 2)
                {
                    double mean = rawOutputs[i];
                    double stddev = rawOutputs[i + 1];

                    double actionSample = Functions.RandomGaussian(mean, stddev);
                    actionBuffer.actions[i / 2] = Math.Clamp(actionSample,-1.0,1.0);
                }
            }
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