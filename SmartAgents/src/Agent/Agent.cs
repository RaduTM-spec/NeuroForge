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
        public BehaviorType behavior = BehaviorType.Inference;

        [SerializeField] private ActorNetwork actorNetwork;
        [SerializeField] private NeuralNetwork criticNetwork;
        [SerializeField] private ExperienceBuffer Memory;

        [Space]
        [Min(1), SerializeField] private int observationSize = 2;
        [SerializeField] private ActionType actionSpace = ActionType.Continuous;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [Space]
        [Min(0)] public int timeHorizon = 180_000;
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
        
        // Input normalization
        double[] mins;
        double[] maxs;


        List<Transform> initialTransforms = new List<Transform>();

        #endregion

        #region Setup
        protected virtual void Awake()
        {
            hp = GetComponent<HyperParameters>();
            
            InitNetworks();
            InitMemory();
            InitBuffers();

            InitSensors(this.transform);
            if(OnEpisodeEnd == OnEpisodeEndType.ResetEnvironment)
                InitInitialTransforms(this.transform.parent);
            else if(OnEpisodeEnd == OnEpisodeEndType.ResetAgent)
                InitInitialTransforms(this.transform);
        }
        private void InitNetworks()
        {
            if (actorNetwork)
            {
                observationSize = actorNetwork.GetObservationsNumber();
                

                if (actorNetwork.actionSpace == ActionType.Continuous)
                {
                    actionSpace = ActionType.Continuous;
                    ContinuousSize = actorNetwork.GetActionsNumber();
                }
                else
                {
                    actionSpace = ActionType.Discrete;
                    DiscreteBranches = actorNetwork.outputShape;
                }
            }

            ActivationType activation = hp.activationType;

            if (actorNetwork == null) actorNetwork = actionSpace == ActionType.Continuous ?
                        new ActorNetwork(observationSize, ContinuousSize, hp.hiddenUnits, hp.layersNumber, activation) :
                        new ActorNetwork(observationSize, DiscreteBranches, hp.hiddenUnits, hp.layersNumber, activation);

            if (criticNetwork == null) criticNetwork = new NeuralNetwork(observationSize, 1, hp.hiddenUnits,hp.layersNumber, activation, ActivationType.Linear, LossType.MeanSquare, true, GetCriticName());
            
            string GetCriticName()
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<NeuralNetwork>("Assets/CriticNN#" + id + ".asset") != null)
                    id++;
                return "CriticNN#" + id;
            }
            
        }
        private void InitMemory()
        {
            if (Memory == null) Memory = new ExperienceBuffer(true);

            Memory.Clear();

            
        }
        private void InitBuffers()
        {
            sensorBuffer = new SensorBuffer(observationSize);
            actionBuffer = new ActionBuffer(actorNetwork.GetActionsNumber());
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
                case BehaviorType.Self:
                    ActiveAction();
                    break;
                case BehaviorType.Inference:
                    InferenceAction();
                    break;
                case BehaviorType.Manual:
                    ManualAction();
                    break;
            }
            Step++;
            if (timeHorizon != 0 && Step >= timeHorizon && behavior == BehaviorType.Inference)
                EndEpisode();
        }
        private void ManualAction()
        {

        }
        private void ActiveAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectSensorsObservations(sensorBuffer);
            CollectObservations(sensorBuffer);
            NormalizeObservations(sensorBuffer);

            if(actionSpace == ActionType.Continuous)
            {
                actionBuffer.continuousActions = actorNetwork.ContinuousForwardPropagation(sensorBuffer.observations).Item2;           
            }
            else
            {
                actionBuffer.discreteActions = actorNetwork.DiscreteForwardPropagation(sensorBuffer.observations).Item2;
            }
            OnActionReceived(actionBuffer);

        }
        private void InferenceAction()
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
                lock (actorNetwork) lock (criticNetwork)
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

            double value = criticNetwork.ForwardPropagation(sensorBuffer.observations)[0];
            double[] log_probs;

            if(actionSpace == ActionType.Continuous)
            {
                (double[], float[]) outs_acts = actorNetwork.ContinuousForwardPropagation(sensorBuffer.observations);
                log_probs = actorNetwork.GetContinuousLogProbs(outs_acts.Item1, outs_acts.Item2);
                actionBuffer.continuousActions = outs_acts.Item2;
                Memory.Store(sensorBuffer.observations, outs_acts.Item1, stepReward, log_probs, value, episodeDone);
            }
            else
            {
                (double[],int[]) dist_acts = actorNetwork.DiscreteForwardPropagation(sensorBuffer.observations);
                log_probs = ActorNetwork.GetDiscreteLogProbs(dist_acts.Item1);
                actionBuffer.discreteActions = dist_acts.Item2;
                Memory.Store(sensorBuffer.observations, dist_acts.Item1, stepReward, log_probs, value, episodeDone);

            }

            stepReward = 0;
       
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
            if(actionSpace == ActionType.Continuous)
            {
                for (int t = 0; t < mini_batch.Count; t++)
                {
                    (double[], float[]) forwardPropagation = actorNetwork.ContinuousForwardPropagation(mini_batch[t].state);

                    double[] old_log_probs = mini_batch[t].log_probs;
                    double[] new_log_probs = actorNetwork.GetContinuousLogProbs(forwardPropagation.Item1, forwardPropagation.Item2);

                    double[] ratios = new double[new_log_probs.Length];
                    for (int r = 0; r < ratios.Length; r++)
                    {
                        ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
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

                    double[] entropies = new double[ratios.Length];
                    for (int e = 0; e < entropies.Length; e++)
                    {
                        double mu = forwardPropagation.Item1[e * 2];
                        double sigma = forwardPropagation.Item1[e * 2 + 1];
                        entropies[e] = 0.5 * Math.Log(2 * Math.PI * Math.E * sigma * sigma);
                    }

                    double[] actor_losses = new double[forwardPropagation.Item1.Length];

                    for (int l = 0; l < actor_losses.Length; l++)
                    {
                        actor_losses[l] = surrogate_losses[l] + hp.entropyRegularization * entropies[l];
                    }    

                    actorNetwork.BackPropagation(mini_batch[t].state, actor_losses);
                    criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });
                }
            }else
            if(actionSpace == ActionType.Discrete)
            {
                for (int t = 0; t < mini_batch.Count; t++)
                {
                    double[] dist = actorNetwork.DiscreteForwardPropagation(mini_batch[t].state).Item1;

                    double[] old_log_probs = mini_batch[t].log_probs;
                    double[] new_log_probs = ActorNetwork.GetDiscreteLogProbs(dist);

                    double[] ratios = new double[new_log_probs.Length];
                    for (int r = 0; r < ratios.Length; r++)
                    {
                        ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
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

                    double[] entropies = new double[ratios.Length];
                    for (int e = 0; e < entropies.Length; e++)
                    {
                        entropies[e] = - new_log_probs[e] * Math.Exp(new_log_probs[e]);
                    }

                    double[] actor_losses = new double[dist.Length];

                    for (int l = 0; l < actor_losses.Length; l++)
                    {
                        actor_losses[l] = surrogate_losses[l] + hp.entropyRegularization * entropies[l];
                    }

                    actorNetwork.BackPropagation(mini_batch[t].state, actor_losses);
                    criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });

                    
                }
            }

            actorNetwork.OptimizeParameters(hp.actorLearnRate, hp.momentum, hp.regularization, false);
            criticNetwork.OptimizeParameters(hp.criticLearnRate, hp.momentum, hp.regularization, true);


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
                if (obs[i] > maxs[i]) //if i place else i remain with NaN value on (max)
                    maxs[i] = obs[i];
            }

            //normalize the obs (-1,1)
            for (int i = 0; i < obs.Length; i++)
            {
                if (maxs[i] == mins[i])
                    continue;

                obs[i] = 2 * (obs[i] - mins[i]) / (maxs[i] - mins[i]) - 1;
            }

        }
        private void NormalizeAdvantages(List<double> advantages)
        {
            //Normalize advantages
            double mean = advantages.Sum() / advantages.Count;
            double std = Math.Sqrt(advantages.Sum(x => Math.Pow(x - mean, 2) / advantages.Count));
            for (int i = 0; i < advantages.Count; i++)
            {
                advantages[i] = (advantages[i] - mean) / (std + 0.00000001);
            }
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