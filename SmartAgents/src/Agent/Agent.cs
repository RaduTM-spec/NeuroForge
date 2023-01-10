using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        #region Visible Fields
        public BehaviorType behavior = BehaviorType.Passive;

        [SerializeField] private ArtificialNeuralNetwork policyNetwork;
        [SerializeField] private ArtificialNeuralNetwork criticNetwork;
        [SerializeField] private ExperienceBuffer Memory;


        [Space, Min(1), SerializeField] private int observationSize = 2;
        
        [SerializeField] private ActionType actionType = ActionType.Continuous;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [Space, Min(0)] public int timeHorizon = 180_000;
        [SerializeField] private OnEpisodeEndType OnEpisodeEnd = OnEpisodeEndType.ResetEnvironment;

        #endregion

        #region Hidden Fields
        private int Episode = 1;
        private int Step = 0;
        private double stepReward = 0;
        private double episodeReward = 0;

        private HyperParameters hp;
        private List<RaySensor> raySensors = new List<RaySensor>();
        private List<CameraSensor> cameraSensors = new List<CameraSensor>();

        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;
        

        //Input normalization
        double[] mins;
        double[] maxs;


        List<Transform> initialTransforms = new List<Transform>();

        #endregion

        #region Setup
        protected virtual void Awake()
        {
            hp = GetComponent<HyperParameters>();
            
            InitNetworks_InitMemory_InitBuffers();
            InitSensors(this.transform);
            if(OnEpisodeEnd == OnEpisodeEndType.ResetEnvironment)
                InitInitialTransforms(this.transform.parent);
            else if(OnEpisodeEnd == OnEpisodeEndType.ResetAgent)
                InitInitialTransforms(this.transform);
        }
        private void InitNetworks_InitMemory_InitBuffers()
        {
            if (policyNetwork)
            {
                observationSize = policyNetwork.GetInputsNumber();
                ContinuousSize = policyNetwork.GetOutputsNumber();
                if (policyNetwork.outputActivationType == ActivationType.SoftMax)
                    actionType = ActionType.Discrete;
                else
                    actionType = ActionType.Continuous;
            }

            sensorBuffer = new SensorBuffer(ContinuousSize);
            actionBuffer = new ActionBuffer(observationSize);

            ActivationType activation = hp.activationType;
            ActivationType outputActivation;

            int[] actorOutputs;

            if (actionType == ActionType.Discrete)
            {
                actorOutputs = DiscreteBranches;
                outputActivation = ActivationType.BranchedSoftMaxActivation;
            }
            else //actionType == ActionType.Continuous
            {
                actorOutputs = new int[] { ContinuousSize };
                outputActivation = ActivationType.PairedTanhSoftPlusActivation;
            }

            if (policyNetwork == null) policyNetwork = new ArtificialNeuralNetwork(observationSize, actorOutputs, hp.HiddenLayerUnits,hp.HiddenLayersNumber, activation, outputActivation, LossType.MeanSquare, true, GetPolicyName());
            if (criticNetwork == null) criticNetwork = new ArtificialNeuralNetwork(observationSize, new int[] {1}, hp.HiddenLayerUnits,hp.HiddenLayersNumber, activation, ActivationType.Tanh, LossType.MeanSquare, true, GetCriticName());
            if (Memory == null) Memory = new ExperienceBuffer(GetMemoryName(), true);

            Memory.Clear();

            string GetPolicyName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/Policy#" + id + ".asset") != null)
                    id++;
                return "PolicyNN#" + id;
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
        private void InitSensors(Transform parent)
        {
            RaySensor rayFound = GetComponent<RaySensor>();
            CameraSensor camFound = GetComponent<CameraSensor>();
            if (rayFound != null && rayFound.enabled)
                raySensors.Add(rayFound);
            if (camFound != null && camFound.enabled)
                cameraSensors.Add(camFound);
            foreach (Transform child in parent)
            {
                InitSensors(child);
            }
        }
        private void InitInitialTransforms(Transform parent)
        {
            foreach (Transform child in parent)
            {
                Transform clone = new GameObject().transform;

                clone.position = child.position;
                clone.rotation = child.rotation;
                clone.localScale = child.localScale;

                initialTransforms.Add(clone);
                InitInitialTransforms(child);
            }
        }

        #endregion

        #region Loop
        protected virtual void Update()
        {
            switch (behavior)
            {
                case BehaviorType.Active:
                    ActiveAction();
                    break;
                case BehaviorType.Inference:
                    LearnAction();
                    break;
            }
            Step++;
            if (timeHorizon != 0 && Step >= timeHorizon && behavior == BehaviorType.Inference)
                EndEpisode();
        }
        private void ActiveAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensorsObservations(sensorBuffer);
            CollectObservations(sensorBuffer);
            NormalizeObservations(sensorBuffer);

            double[] rawOutput = policyNetwork.ForwardPropagation(sensorBuffer.observations);

            // Continuous output form: (mean, stddev) (mean, stddev) (mean, stddev) ...
            if (actionType == ActionType.Continuous)
                for (int i = 0; i < rawOutput.Length; i += 2)
                {
                    double mean = rawOutput[i];
                    double stddev = rawOutput[i+ 1];

                    actionBuffer.continuousActions[i / 2] = (float)Math.Clamp(Functions.RandomGaussian(mean, stddev),-1.0,1.0); 
                }

            OnActionReceived(actionBuffer);

        }
        private void LearnAction()
        {
            Collect_Action_Store(false);

            if (!Memory.IsFull(hp.buffer_size))
                return;
            
            var ret_and_adv = GAE();

            for (int i = 0; i < hp.buffer_size / hp.batch_size; i++)
            {
                List<Sample> miniBatch = Memory.records.GetRange(i, i + hp.batch_size);
                List<double> returns = ret_and_adv.Item1.GetRange(i, i + hp.batch_size);
                List<double> advantages = ret_and_adv.Item2.GetRange(i, i + hp.batch_size);
                lock (policyNetwork) lock (criticNetwork)
                {
                    UpdateActorCritic(miniBatch, returns, advantages);
                }
            }

            Memory.Clear();   
        }

        #endregion

        #region PPO Training
        private void Collect_Action_Store(bool episodeDone)
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensorsObservations(sensorBuffer);
            CollectObservations(sensorBuffer);
            NormalizeObservations(sensorBuffer);

            double[] rawOutputs = policyNetwork.ForwardPropagation(sensorBuffer.observations);

            //Discrete output form: 1branch(x,y,z) x+y+z = 1 (softmax)
            //Continous output form: (mean, stddev) (mean, stddev) (mean, stddev) ...

            double[] log_probs = GetLogProbs(rawOutputs);
            double value = criticNetwork.ForwardPropagation(sensorBuffer.observations)[0];

            Memory.Store(sensorBuffer.observations, rawOutputs, stepReward, log_probs, value, episodeDone);
            stepReward = 0;

            Fill_ActionBuffer(rawOutputs);
            OnActionReceived(actionBuffer);
        }
        private (List<double>, List<double>) GAE()
        {
            List<Sample> iteration_data = Memory.records;

            //returns = discounted rewards
            List<double> returns = new List<double>();
            List<double> advantages = new List<double>();

            //Normalize rewards 
            double minReward = iteration_data.Min(s => s.reward);
            double maxReward = iteration_data.Max(s => s.reward);
            for(int i = 0; i < iteration_data.Count; i++)
            {
                if (iteration_data[i].reward == 0)
                    continue;
                if (iteration_data[i].reward < 0)
                    iteration_data[i].reward /= -minReward;
                else
                    iteration_data[i].reward /= maxReward;
            }

            //Calculate returns and advantages
            double advantage = 0;
            for (int i = iteration_data.Count - 1; i >= 0; i--)
            {
                double value = criticNetwork.ForwardPropagation(iteration_data[i].state)[0];
                double nextValue = i == iteration_data.Count-1? 
                                        0 : 
                                        criticNetwork.ForwardPropagation(iteration_data[i + 1].state)[0];


                double delta = iteration_data[i].done? 
                               iteration_data[i].reward - value :
                               iteration_data[i].reward + hp.discountFactor * nextValue - value;

                advantage = advantage * hp.discountFactor * hp.gaeFactor + delta;

                advantages.Add(advantage);
                returns.Add(advantage + value);
            }
            NormalizeAdvantages(advantages);

            advantages.Reverse();           
            returns.Reverse();

            return (returns,advantages);
        }
        private void UpdateActorCritic(List<Sample> mini_batch, List<double> mb_returns, List<double> mb_advantages)
        {
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
                                             ratios[r] * mb_advantages[t],
                                             Math.Clamp(ratios[r], 1 - hp.clipFactor, 1 + hp.clipFactor) * mb_advantages[t]
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


                criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });
                
                //SURROGATE LOSS
                double[] actor_loss = new double[rawOutput.Length];
                for (int i = 0; i < actor_loss.Length; i++)
                {
                    actor_loss[i] = surrogate_losses[i/2] + hp.entropyRegularization * entropies[i/2];

                }               
                policyNetwork.BackPropagation_LossCalculated(mini_batch[t].state, actor_loss);
               
            }

            criticNetwork.OptimizeParameters(hp.learnRate, hp.momentum, hp.regularization, true);  // Gradient descent
            policyNetwork.OptimizeParameters(hp.learnRate, hp.momentum, hp.regularization, false); // Gradient ascent

        }

        private void NormalizeObservations(SensorBuffer sensorBuff)
        {
            double[] obs = sensorBuff.observations;

            //Init
            if (mins == null || maxs == null)
            {
                mins = new double[obs.Length];
                maxs = new double[obs.Length];
                for (int i = 0; i < obs.Length; i++)
                {
                    mins[i] = double.MaxValue;
                    maxs[i] = double.MinValue;
                }
            }

            //Find new min or max
            for (int i = 0; i < obs.Length; i++)
            {
                if (obs[i] < mins[i])
                    mins[i] = obs[i];
                else if (obs[i] > maxs[i])
                    maxs[i] = obs[i];
            }

            //normalize the obs (-1,1)
            for (int i = 0; i < obs.Length; i++)
            {
                obs[i] = 2 * (obs[i] - mins[i]) / (maxs[i] - mins[i]) - 1;
            }

        }
        void NormalizeAdvantages(List<double> advantages)
        {
            //Normalize advantages
            double mean = advantages.Sum() / advantages.Count;
            double std = Math.Sqrt(advantages.Sum(x => Math.Pow(x - mean, 2) / advantages.Count));
            for (int i = 0; i < advantages.Count; i++)
            {
                advantages[i] = (advantages[i] - mean) / (std + 0.00000001);
            }
        }
        double[] GetLogProbs(double[] rawOutputs)
        {
            List<double> log_probs = new List<double>();
            if (actionType == ActionType.Discrete)
            {
                for (int i = 0; i < rawOutputs.Length; i++)
                {
                    log_probs.Add(Math.Log(rawOutputs[i]));
                }
            }
            else // actionType == ActionType.Continuous
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
        void Fill_ActionBuffer(double[] rawOutputs)
        {
            if (actionType == ActionType.Discrete)
            {
                actionBuffer.discreteActions = new int[] { DiscreteBranches.Length};

                for (int branchNo = 0; branchNo < DiscreteBranches.Length; branchNo++)
                {
                    double max = double.MinValue;
                    int index = -1;
                    for (int branchElem = 0; branchElem < DiscreteBranches[branchNo]; branchElem++)
                    {
                        if (rawOutputs[branchNo + branchElem] > max)
                        {
                            max = rawOutputs[branchNo + branchElem];
                            index = branchElem;
                        } 
                    }
                    actionBuffer.discreteActions[branchNo] = index;
                }
            }
            else if (actionType == ActionType.Continuous)
            {
                for (int i = 0; i < rawOutputs.Length; i += 2)
                {
                    double mean = rawOutputs[i];
                    double stddev = rawOutputs[i + 1];

                    double actionSample = Functions.RandomGaussian(mean, stddev);
                    actionBuffer.continuousActions[i / 2] = (float)Math.Clamp(actionSample,-1.0,1.0);
                }
            }
        }
        public int DecideDiscreteBranchAction(double[] rawBranchOutputs)
        {
            //discreteActions will contain a the highest probable action from a branch
            int index = -1;
            double max = double.MinValue;
            for (int i = 0; i < rawBranchOutputs.Length; i++)
                if (rawBranchOutputs[i] > max)
                {
                    max = rawBranchOutputs[i];
                    index = i;
                }
            return index;
        }

        #endregion

        #region Utils     
        private void CollectSensorsObservations(SensorBuffer buffer)
        {
            foreach (var raySensor in raySensors)
            {
                buffer.AddObservation(raySensor.GetObservations());
            }
            foreach (var camSensor in cameraSensors)
            {
                buffer.AddObservation(camSensor.FlatCapture());
            }
        }
        private void ResetToInitialTransforms(Transform parent, ref int index)
        {
            for (int i = 0; i < parent.transform.childCount; i++)
            {
                Transform child = parent.transform.GetChild(i);
                Transform initialTransform = initialTransforms[index++];

                child.position = initialTransform.position;
                child.rotation = initialTransform.rotation;
                child.localScale = initialTransform.localScale;

                ResetToInitialTransforms(child, ref index);
            }
        }

        // Used by the User
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
            this.stepReward += Convert.ToDouble(reward);
            this.episodeReward += stepReward;
        }
        public void EndEpisode()
        {
            // Collect last data piece (including the terminal reward)
            Collect_Action_Store(true);

            int transformsStart = 0;
            if (OnEpisodeEnd == OnEpisodeEndType.ResetEnvironment)
                ResetToInitialTransforms(this.transform.parent, ref transformsStart);
            else if (OnEpisodeEnd == OnEpisodeEndType.ResetAgent)
                ResetToInitialTransforms(this.transform, ref transformsStart);

            OnEpisodeBegin();
            PrintEpsiodeStatistic();

            Episode++;
            Step = 0;
            episodeReward = 0;


            void PrintEpsiodeStatistic()
            {
                StringBuilder statistic = new StringBuilder();
                statistic.Append("Episode: ");
                statistic.Append(Episode);
                statistic.Append(" | Steps: ");
                statistic.Append(Step);
                statistic.Append(" | Cumulated Reward: ");
                statistic.Append(episodeReward);
                UnityEngine.Debug.Log(statistic.ToString());
            }
        }

        #endregion
    }
    #region Custom Editor
    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    class ScriptlessAgent : Editor
    {
        public override void OnInspectorGUI()
        {
            SerializedProperty actType = serializedObject.FindProperty("actionType");
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