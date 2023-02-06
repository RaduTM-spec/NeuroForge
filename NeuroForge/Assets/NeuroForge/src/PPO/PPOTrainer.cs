using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.Threading.Tasks;
using System.IO;
using Unity.VisualScripting;

namespace NeuroForge
{
    public sealed class PPOTrainer : MonoBehaviour
    {
        public static PPOTrainer Instance;

        [SerializeField] private List<PPOAgent> agents;
        private PPONetwork model;
        private PPOHyperParameters hp;
        private ActionType actionSpace;
        private int agentsReady = 0;

        private void Awake()
        {
            if(Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
            }
        }
        private void LateUpdate()
        {
            if (Instance.agentsReady == Instance.agents.Count)
            {
                Debug.Log("Learn");
                Train();
            }
        }

        public static void Subscribe(PPOAgent mainAgent)
        {
            if (Instance == null)
            {
                GameObject go = new GameObject("PPOTrainer");
                go.AddComponent<PPOTrainer>();
                Instance.agents = new List<PPOAgent>();
                Instance.model = mainAgent.model;
                Instance.hp = mainAgent.hp;
                Instance.actionSpace = mainAgent.GetActionSpace();        
            }
            
            // Subscribe
            Instance.agents.Add(mainAgent);
        }
        public static void Ready()
        {
            Instance.agentsReady++;
        }
        private void Train()
        {
            foreach (var agent in Instance.agents)
            {
                List<PPOSample> playback = agent.Memory.records;
                var pair = GAE(playback, Instance.hp.discountFactor);
                List<double> advantages = pair.Item1;
                List<double> returns = pair.Item2;
                

                using (StreamWriter sw = new StreamWriter("C:\\Users\\X\\Desktop\\debug.txt", true))
                {
                    sw.WriteLine("\n\n\n--------------> Memory: " + hp.buffer_size + " <----------------");
                    for (int i = 0; i < advantages.Count; i++)
                    {
                        sw.Write("[ reward: " + playback[i].reward.ToString("00.00") + " ]");
                        sw.Write("[ value: " + playback[i].value.ToString("00.00") + " ]");
                        sw.Write("[ done: " + playback[i].done + " ]");
                        sw.Write("[ return: " + returns[i].ToString("00.00" + " ]"));
                        sw.Write("[ advantage: " + advantages[i].ToString("00.00") + " ]\n");
                    }
                }

                for (int i = 0; i < Instance.hp.buffer_size / Instance.hp.batch_size; i++)
                {
                    int posInBuffer = i * Instance.hp.batch_size;

                    List<PPOSample> miniBatch = new List<PPOSample>(playback.GetRange(posInBuffer, Instance.hp.batch_size));
                    List<double> miniBatch_advantages = new List<double>(advantages.GetRange(posInBuffer, Instance.hp.batch_size));
                    List<double> miniBatch_returns = new List<double>(returns.GetRange(posInBuffer, Instance.hp.batch_size));

                    if (Instance.actionSpace == ActionType.Continuous)
                        UpdateContinuousModel(miniBatch, miniBatch_advantages, miniBatch_returns);
                    else
                        UpdateDiscreteModel(miniBatch, miniBatch_advantages, miniBatch_returns);
                }
                

                agent.Memory.Clear();
                Instance.agentsReady--;
            }
        }


        void UpdateDiscreteModel(List<PPOSample> mini_batch, List<double> mb_advantages, List<double> mb_returns)
        {
            for (int t = 0; t < mini_batch.Count; t++)
            {
                double[] distributions = Instance.model.actorNetwork.DiscreteForwardPropagation(mini_batch[t].state).Item1;

                double[] old_log_probs = mini_batch[t].log_probs;
                double[] new_log_probs = PPOActorNetwork.GetDiscreteLogProbs(distributions);

                // Calculate ratios
                double[] ratios = new double[new_log_probs.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
                }

                // Calculate surrogate loss
                double[] clipped_surrogate_objective = new double[ratios.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    // gradient ascent, so loss *= -1
                    clipped_surrogate_objective[r] = -Math.Min
                                                     (
                                                           ratios[r] * mb_advantages[t],
                                                           Math.Clamp(ratios[r], 1.0 - Instance.hp.clipFactor, 1.0 + Instance.hp.clipFactor) * mb_advantages[t]
                                                     );
                }

                // Add entropy
                for (int i = 0; i < clipped_surrogate_objective.Length; i++)
                {
                    double entropy = -distributions[i] * new_log_probs[i]; // phi log phi
                    clipped_surrogate_objective[i] -= entropy * Instance.hp.entropyRegularization;
                }

                // Update policy with SGD
                Instance.model.actorNetwork.BackPropagation(mini_batch[t].state, clipped_surrogate_objective);
                Instance.model.criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });

                Instance.model.actorNetwork.OptimizeParameters(Instance.hp.actorLearnRate, Instance.hp.momentum, Instance.hp.regularization);
                Instance.model.criticNetwork.OptimizeParameters(Instance.hp.criticLearnRate, Instance.hp.momentum, Instance.hp.regularization);

            }
        }
        void UpdateContinuousModel(List<PPOSample> mini_batch, List<double> mb_advantages, List<double> mb_returns)
        {
            for (int t = 0; t < mini_batch.Count; t++)
            {
                (double[], float[]) forwardPropagation = Instance.model.actorNetwork.ContinuousForwardPropagation(mini_batch[t].state);

                double[] old_log_probs = mini_batch[t].log_probs;
                double[] new_log_probs = Instance.model.actorNetwork.GetContinuousLogProbs(forwardPropagation.Item1, forwardPropagation.Item2);

                // Calculate ratios
                double[] ratios = new double[new_log_probs.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
                }

                // Calculate surroagate loss
                double[] clipped_surrogate_objective = new double[ratios.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    // gradient ascent, so loss *= -1
                    clipped_surrogate_objective[r] = -Math.Min
                                                     (
                                                           ratios[r] * mb_advantages[t],
                                                           Math.Clamp(ratios[r], 1.0 - Instance.hp.clipFactor, 1.0 + Instance.hp.clipFactor) * mb_advantages[t]
                                                     );
                }

                // Calculate entropies
                double[] entropies = new double[ratios.Length];
                for (int e = 0; e < entropies.Length; e += 2)
                {
                    double sigma = forwardPropagation.Item2[e + 1];
                    double entropy = Math.Sqrt(2 * Math.PI * Math.E * sigma * sigma);

                    entropies[e] = entropy;
                    entropies[e + 1] = entropy;
                }

                // Apply entropies
                for (int i = 0; i < clipped_surrogate_objective.Length; i++)
                {
                    clipped_surrogate_objective[i] -= entropies[i] * Instance.hp.entropyRegularization;
                }


                // Update policy SGD 
                Instance.model.actorNetwork.BackPropagation(mini_batch[t].state, clipped_surrogate_objective);
                Instance.model.criticNetwork.BackPropagation(mini_batch[t].state, new double[] { mb_returns[t] });

                Instance.model.actorNetwork.OptimizeParameters(Instance.hp.actorLearnRate, Instance.hp.momentum, Instance.hp.regularization);
                Instance.model.criticNetwork.OptimizeParameters(Instance.hp.criticLearnRate, Instance.hp.momentum, Instance.hp.regularization);
            }
        }
        (List<double>, List<double>) DeprecatedGAE(List<PPOSample> playback, float gamma, float lambda)
        {
            List<double> advantages = new List<double>();
            List<double> returns = new List<double>();

            double runningAdvantage = 0;
            double nextValue = 0;

            for (int i = playback.Count - 1; i >= 0; i--)
            {
                double mask = playback[i].done ? 0 : 1;
                double delta = playback[i].reward + nextValue * gamma * mask - playback[i].value;

                runningAdvantage = delta + runningAdvantage * gamma * lambda; // * mask?
                nextValue = playback[i].done ? 0 : playback[i].value;

                advantages.Insert(0, runningAdvantage);
                returns.Insert(0, runningAdvantage + playback[i].value);
            }

            if (Instance.hp.normalizeAdvantages)
                Functions.Normalize(advantages);

            return (advantages,returns);
        }
        (List<double>, List<double>) GAE(List<PPOSample> playback, float gamma)
        {
            List<double> advantages = new List<double>();
            List<double> returns = new List<double>();

            for (int i = 0; i < playback.Count; i++)
            {
                double discountedReturn = 0;
                double discount = 1;
                for (int j = i; j < playback.Count; j++)
                {
                    discountedReturn += playback[j].reward * discount;
                    discount *= gamma;
                }
                returns.Add(discountedReturn);
                advantages.Add(discountedReturn - playback[i].value);
            }

            if (Instance.hp.normalizeAdvantages)
                Functions.Normalize(advantages);

            return (advantages, returns);
        }


        void ShuffleTrainingData(List<PPOSample> playback, List<double> advantages, List<double> returns)
        {
            // Not necesarry for now
        }
    }
}