using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        #region Visible Fields
        public BehaviorType behavior = BehaviorType.Inference;

        [SerializeField] public PPOModel model;
        [HideInInspector] public PPOMemory Memory;

        [Space]
        [Min(1), SerializeField] private int observationSize = 2;
                [SerializeField] private ActionType actionSpace = ActionType.Continuous;
        [Min(1), SerializeField] private int ContinuousSize;
        [Min(1), SerializeField] private int[] DiscreteBranches;

        [Space]
        [Min(0), Tooltip("Episode max steps")] public int timeHorizon = 180_000;
        [SerializeField] private OnEpisodeEndType OnEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        
        #endregion

        #region Hidden Fields
        private int Episode = 1;
        private int Step = 0;
        private double stepReward = 0;
        private double episodeReward = 0;

        [HideInInspector] public HyperParameters hp;
        private List<RaySensor> raySensors = new List<RaySensor>();
        private List<CameraSensor> cameraSensors = new List<CameraSensor>();

        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;

        List<Transform> initialTransforms = new List<Transform>();

        #endregion

        #region Setup
        protected virtual void Awake()
        {
            hp = GetComponent<HyperParameters>();

            // Init everything
            InitNetwork();
           
            // Init memory
            Memory = new PPOMemory(true);
           
            // Init buffers
            sensorBuffer = new SensorBuffer(observationSize);
            actionBuffer = new ActionBuffer(model.actorNetwork.GetActionsNumber());
            
            // Init sensors
            InitSensors(this.transform);

            // Init environment
            if(OnEpisodeEnd == OnEpisodeEndType.ResetEnvironment)
                InitInitialTransforms(this.transform.parent);
            else if(OnEpisodeEnd == OnEpisodeEndType.ResetAgentOnly)
                InitInitialTransforms(this.transform);
        }
        private void InitNetwork()
        {
            if (model)
            {
                observationSize = model.actorNetwork.GetObservationsNumber();
                
                if (model.actorNetwork.actionSpace == ActionType.Continuous)
                {
                    actionSpace = ActionType.Continuous;
                    ContinuousSize = model.actorNetwork.GetActionsNumber();
                }
                else
                {
                    actionSpace = ActionType.Discrete;
                    DiscreteBranches = model.actorNetwork.outputBranches;
                }
            }

            if(!model)
            {
                model = actionSpace == ActionType.Continuous ?
                        new PPOModel(observationSize, ContinuousSize, hp.hiddenUnits, hp.layersNumber, hp.activationType, hp.initializationType) :
                        new PPOModel(observationSize, DiscreteBranches, hp.hiddenUnits, hp.layersNumber, hp.activationType, hp.initializationType);
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
                case BehaviorType.Self:
                    ActiveAction();
                    break;
                case BehaviorType.Inference:
                    Collect_Action_Store(false);
                    break;
                case BehaviorType.Manual:
                    ManualAction();
                    break;
            }
            Step++;
            if (timeHorizon != 0 && Step >= timeHorizon && behavior == BehaviorType.Inference)
                EndEpisode();
        }
        protected virtual void LateUpdate()
        {
            if (behavior == BehaviorType.Inference && Memory.IsFull(hp.buffer_size))
                UpdatePolicy();
        }
        private void ManualAction()
        {

        }
        private void ActiveAction()
        {
            sensorBuffer.Clear();
            actionBuffer.Clear();

            CollectObservations(sensorBuffer);
            CollectSensors(sensorBuffer);
            
            model.observationsNormalizer.Normalize(sensorBuffer.observations);

            if(actionSpace == ActionType.Continuous)
            {
                actionBuffer.continuousActions = model.actorNetwork.ContinuousForwardPropagation(sensorBuffer.observations).Item2;           
            }
            else
            {
                actionBuffer.discreteActions = model.actorNetwork.DiscreteForwardPropagation(sensorBuffer.observations).Item2;
            }
            OnActionReceived(actionBuffer);

        }
        private void UpdatePolicy()
        {
            var ret_and_adv = GAE();

            for (int i = 0; i < hp.buffer_size / hp.batch_size; i++)
            {
                int lower_bound = i * hp.batch_size;
                int upper_bound = hp.batch_size;
                List<PPOSample> miniBatch = new List<PPOSample>(Memory.records.GetRange(lower_bound, upper_bound));
                List<double> returns = new List<double>(ret_and_adv.Item1.GetRange(lower_bound, upper_bound));
                List<double> advantages = new List<double>(ret_and_adv.Item2.GetRange(lower_bound, upper_bound));

                lock (model.actorNetwork) lock (model.criticNetwork)
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

            CollectObservations(sensorBuffer);
            CollectSensors(sensorBuffer);
            
            model.observationsNormalizer.Normalize(sensorBuffer.observations);

            double value = model.criticNetwork.ForwardPropagation(sensorBuffer.observations)[0];

            double[] log_probs;

            if(actionSpace == ActionType.Continuous)
            {
                (double[], float[]) outs_acts = model.actorNetwork.ContinuousForwardPropagation(sensorBuffer.observations);
                log_probs = model.actorNetwork.GetContinuousLogProbs(outs_acts.Item1, outs_acts.Item2);
                actionBuffer.continuousActions = outs_acts.Item2;
                Memory.Store(sensorBuffer.observations, outs_acts.Item1, stepReward, log_probs, value, episodeDone);
            }
            else
            {
                (double[],int[]) dist_acts = model.actorNetwork.DiscreteForwardPropagation(sensorBuffer.observations);
                log_probs = ActorNetwork.GetDiscreteLogProbs(dist_acts.Item1);
                actionBuffer.discreteActions = dist_acts.Item2;
                Memory.Store(sensorBuffer.observations, dist_acts.Item1, stepReward, log_probs, value, episodeDone);

            }

            stepReward = 0;
       
            OnActionReceived(actionBuffer);   
        }
        private (List<double>, List<double>) GAE()
        {
            List<PPOSample> playback = Memory.records;

            //returns = discounted rewards
            List<double> returns = new List<double>();
            List<double> advantages = new List<double>();


            //Normalize rewards 
            double mean = playback.Average(t => t.reward);
            double std = Math.Sqrt(playback.Sum(t => (t.reward - mean) * (t.reward - mean) / playback.Count)) ;
            if (std == 0) std = +1e-8;
            IEnumerable<double> normalizedRewards = playback.Select(r => (r.reward - mean) / std);


            //Calculate returns and advantages
            double runningAdvantage = 0;
            for (int i = playback.Count - 1; i >= 0; i--)
            {
                double value = playback[i].value;
                double nextValue = i == playback.Count - 1?
                                        0 :
                                        playback[i+1].value;

                double mask = playback[i].done ? 0 : 1;
                double delta = playback[i].reward + hp.discountFactor * nextValue * mask - value;

                runningAdvantage = hp.discountFactor * hp.gaeFactor * runningAdvantage + delta;

                returns.Insert(0, runningAdvantage + value);
                advantages.Insert(0, model.advantagesNormalizer.Normalize(runningAdvantage));
            }

            return (returns,advantages);
        }
        private void UpdateActorCritic(List<PPOSample> mini_batch, List<double> mb_returns, List<double> mb_advantages)
        {
           /* StringBuilder tuples = new StringBuilder();
            for (int i = 0; i < mini_batch.Count; i++)
            {
                string str = "[ reward: ";
                str += mini_batch[i].reward + " | return: ";
                str += mb_returns[i] + " | advantage: ";
                str += mb_advantages[i] + "] ";
                tuples.Append(str);
            }
            UnityEngine.Debug.Log(tuples.ToString());*/

            if (actionSpace == ActionType.Continuous)
            {
                for (int t = 0; t < mini_batch.Count; t++)
                {
                    (double[], float[]) forwardPropagation = model.actorNetwork.ContinuousForwardPropagation(mini_batch[t].state);

                    double[] old_log_probs = mini_batch[t].log_probs;
                    double[] new_log_probs = model.actorNetwork.GetContinuousLogProbs(forwardPropagation.Item1, forwardPropagation.Item2);

                    double[] ratios = new double[new_log_probs.Length];
                    for (int r = 0; r < ratios.Length; r++)
                    {
                        ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
                    }
                    
                    double[] entropies = new double[ratios.Length];
                    for (int e = 0; e < entropies.Length; e+=2)
                    {
                        double sigma = forwardPropagation.Item2[e + 1];
                        double entropy = Math.Sqrt(2 * Math.PI * Math.E * sigma * sigma);

                        entropies[e] = entropy;
                        entropies[e + 1] = entropy;
                    }

                    double[] surrogate_loss = new double[ratios.Length];
                    for (int r = 0; r < ratios.Length; r++)
                    {
                        double entropy = hp.entropyRegularization * entropies[r];
                        surrogate_loss[r] = -Math.Min
                                           (
                                                 ratios[r] * mb_advantages[t] + entropy,
                                                 Math.Clamp(ratios[r], 1 - hp.clipFactor, 1 + hp.clipFactor) * mb_advantages[t] + entropy
                                           );
                    }

                    model.actorNetwork.ZeroGradients();
                    model.criticNetwork.ZeroGradients();

                    model.actorNetwork.BackPropagation(mini_batch[t].state, surrogate_loss);
                    model.criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });

                    model.actorNetwork.OptimizeParameters(hp.actorLearnRate, hp.momentum, hp.regularization, false);
                    model.criticNetwork.OptimizeParameters(hp.criticLearnRate, hp.momentum, hp.regularization, true);
                }
            }else
            if(actionSpace == ActionType.Discrete)
            {
                for (int t = 0; t < mini_batch.Count; t++)
                {
                    double[] old_log_probs = mini_batch[t].log_probs;
                    double[] dist = model.actorNetwork.DiscreteForwardPropagation(mini_batch[t].state).Item1;
                    double[] new_log_probs = ActorNetwork.GetDiscreteLogProbs(dist);

                    double[] ratios = new double[new_log_probs.Length];
                    for (int r = 0; r < ratios.Length; r++)
                    {
                        ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
                    }
                    
                    double[] entropies = new double[ratios.Length];
                    for (int e = 0; e < entropies.Length; e++)
                    {
                        entropies[e] = -dist[e] * new_log_probs[e];
                    }

                    double[] surrogate_loss = new double[ratios.Length];
                    for (int r = 0; r < ratios.Length; r++)
                    {
                        double entropy = hp.entropyRegularization * entropies[r];
                        surrogate_loss[r] = -Math.Min
                                           (
                                                 ratios[r] * mb_advantages[t] + entropy,
                                                 Math.Clamp(ratios[r], 1 - hp.clipFactor, 1 + hp.clipFactor) * mb_advantages[t] + entropy
                                           );
                    }

                    // Surrogate losses become too small -> it destroys the gradients and i get nan dists

                    model.actorNetwork.ZeroGradients();
                    model.criticNetwork.ZeroGradients();

                    model.actorNetwork.BackPropagation(mini_batch[t].state, surrogate_loss);
                    model.criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });
                    
                    model.actorNetwork.OptimizeParameters(hp.actorLearnRate, hp.momentum, hp.regularization,  false);
                    model.criticNetwork.OptimizeParameters(hp.criticLearnRate, hp.momentum, hp.regularization, true);

                }
            }
        }

        #endregion

        #region Utils     
        private void CollectSensors(SensorBuffer buffer)
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

            // Reset Transforms
            int transformsStart = 0;
            if (OnEpisodeEnd == OnEpisodeEndType.ResetEnvironment)
                ResetToInitialTransforms(this.transform.parent, ref transformsStart);
            else if (OnEpisodeEnd == OnEpisodeEndType.ResetAgentOnly)
                ResetToInitialTransforms(this.transform, ref transformsStart);

            OnEpisodeBegin();

            // Print statistics
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Episode: ");
            statistic.Append(Episode);
            statistic.Append(" | Steps: ");
            statistic.Append(Step);
            statistic.Append(" | Cumulated Reward: ");
            statistic.Append(episodeReward);
            UnityEngine.Debug.Log(statistic.ToString());

            Episode++;
            Step = 0;
            episodeReward = 0;    
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
