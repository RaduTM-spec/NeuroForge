using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.Threading.Tasks;
using System.IO;
using Unity.VisualScripting;
using UnityEditor;

namespace NeuroForge
{
    public sealed class PPOTrainer : MonoBehaviour
    {
        public static PPOTrainer Instance;

        [SerializeField] private List<PPOAgent> agents;
        private PPOActor actorNetwork;
        private NeuralNetwork criticNetwork;
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
                Instance.Train();
            }
        }

        public static void Subscribe(PPOAgent mainAgent)
        {
            if (Instance == null)
            {
                GameObject go = new GameObject("PPOTrainer");
                go.AddComponent<PPOTrainer>();
                Instance.agents = new List<PPOAgent>();
                Instance.actorNetwork = mainAgent.actor;
                Instance.criticNetwork = mainAgent.critic;
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
            TrainingData trainData = new TrainingData();

            // Calculate GAE for each agent playback
            foreach (var agent in Instance.agents)
            {
                List<double> advantages;
                List<double> returns;
                GAE(agent.memory.records, out advantages, out returns);

                trainData.playback.AddRange(agent.memory.records);
                trainData.advantages.AddRange(advantages);
                trainData.returns.AddRange(returns);
            }
           
            // Train for K epochs
            for (int k = 0; k < Instance.hp.epochs; k++)
            {
                // Suffle the training data
                ShuffleTrainingData(trainData);

                // Create mini-batches
                int no_mini_batches = (Instance.hp.buffer_size * agents.Count) / Instance.hp.batch_size;
                for (int mb = 0; mb < no_mini_batches; mb++)
                {
                    int posInBuffer = mb * Instance.hp.batch_size;

                    var miniBatch_playback = new List<PPOSample>(trainData.playback.GetRange(posInBuffer, Instance.hp.batch_size));
                    var miniBatch_advantages = new List<double>(trainData.advantages.GetRange(posInBuffer, Instance.hp.batch_size));
                    var miniBatch_returns = new List<double>(trainData.returns.GetRange(posInBuffer, Instance.hp.batch_size));

                    if (Instance.actionSpace == ActionType.Continuous)
                        UpdateContinuousModel(miniBatch_playback, miniBatch_advantages, miniBatch_returns);
                    else
                        UpdateDiscreteModel(miniBatch_playback, miniBatch_advantages, miniBatch_returns);
                }
                
            }

            // Clear memories
            agents.ForEach(x => x.memory.Clear());
            Instance.agentsReady = 0;
        }


        void UpdateDiscreteModel(List<PPOSample> mb_playback, List<double> mb_advantages, List<double> mb_returns)
        {
            for (int t = 0; t < mb_playback.Count; t++)
            {
                double[] distributions = Instance.actorNetwork.DiscreteForwardPropagation(mb_playback[t].state).Item1;

                double[] old_log_probs = mb_playback[t].log_probs;
                double[] new_log_probs = PPOActor.GetDiscreteLogProbs(distributions);

                // Calculate ratios
                double[] ratios = new double[new_log_probs.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
                }

                // Calculate surrogate loss
                double[] L_CLIP = new double[ratios.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    L_CLIP[r] = -Math.Min
                                (
                                      ratios[r] * mb_advantages[t],
                                      Math.Clamp(ratios[r], 1.0 - Instance.hp.clipFactor, 1.0 + Instance.hp.clipFactor) * mb_advantages[t]
                                );
                }

                // Add entropy
                for (int i = 0; i < L_CLIP.Length; i++)
                {
                    double L_H = -distributions[i] * new_log_probs[i]; // phi log phi (phi = aforementioned parameterization or output)
                    L_CLIP[i] -= L_H * Instance.hp.entropyRegularization;
                }

                // critic label (not loss)
                double[] L_V = new double[] { mb_returns[t] };

                // Update policy with SGD
                Instance.actorNetwork.BackPropagation(mb_playback[t].state, L_CLIP);
                Instance.criticNetwork.BackPropagation(mb_playback[t].state, L_V);

                Instance.actorNetwork.GradientsClipNorm(hp.maxGradNorm);
                Instance.criticNetwork.GradientsClipNorm(hp.maxGradNorm);

                Instance.actorNetwork.OptimiseParameters(Instance.hp.actorLearnRate, Instance.hp.momentum, Instance.hp.regularization);
                Instance.criticNetwork.OptimiseParameters(Instance.hp.criticLearnRate, Instance.hp.momentum, Instance.hp.regularization);

            }
        }
        void UpdateContinuousModel(List<PPOSample> mini_batch, List<double> mb_advantages, List<double> mb_returns)
        {
            for (int t = 0; t < mini_batch.Count; t++)
            {
                (double[], float[]) forwardPropagation = Instance.actorNetwork.ContinuousForwardPropagation(mini_batch[t].state);

                double[] old_log_probs = mini_batch[t].log_probs;
                double[] new_log_probs = Instance.actorNetwork.GetContinuousLogProbs(forwardPropagation.Item1, forwardPropagation.Item2);

                // Calculate ratios
                double[] ratios = new double[new_log_probs.Length];
                for (int r = 0; r < ratios.Length; r++)
                {
                    ratios[r] = Math.Exp(new_log_probs[r] - old_log_probs[r]);
                }
                
                // Calculate surroagate loss
                double[] L_CLIP = new double[ratios.Length];       
                for (int r = 0; r < ratios.Length; r++)
                {
                    L_CLIP[r] = -Math.Min
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
                for (int i = 0; i < L_CLIP.Length; i++)
                {
                    double L_H = entropies[i] * Instance.hp.entropyRegularization;
                    L_CLIP[i] -= L_H;
                }

                // critic label (not loss)
                double[] L_V = new double[] { mb_returns[t] };

                // Update policy SGD 
                Instance.actorNetwork.BackPropagation(mini_batch[t].state, L_CLIP);
                Instance.criticNetwork.BackPropagation(mini_batch[t].state, L_V);

                Instance.actorNetwork.GradientsClipNorm(hp.maxGradNorm);
                Instance.criticNetwork.GradientsClipNorm(hp.maxGradNorm);

                Instance.actorNetwork.OptimiseParameters(Instance.hp.actorLearnRate, Instance.hp.momentum, Instance.hp.regularization);
                Instance.criticNetwork.OptimiseParameters(Instance.hp.criticLearnRate, Instance.hp.momentum, Instance.hp.regularization);
            }
        }

        void TD(List<PPOSample> playback, out List<double> advantages, out List<double> returns)
        {
            float gamma = Instance.hp.discountFactor;

            returns = new List<double>();
            advantages = new List<double>();

            for (int i = 0; i < playback.Count; i++)
            {
                double discount = 1;
                double Vt = 0;

                for (int j = i; j < playback.Count; j++)
                {
                    if (j == playback.Count - 1)
                    {
                        Vt += discount * playback[j].value;
                    }
                    else
                    {
                        Vt += playback[j].reward * discount;
                        discount *= gamma;

                        if (playback[i].done)
                            break;
                    }  
                }

                returns.Add(Vt);
                advantages.Add(Vt - playback[i].value);
            }

        }
        void GAE(List<PPOSample> playback, out List<double> advantages, out List<double> returns)
        {
            double gamma = Instance.hp.discountFactor;
            double lambda = Instance.hp.gaeFactor;

            returns = new List<double>();
            advantages = new List<double>();

            double Vt = 0;
            double At = 0;
            double nextValue = 0;

            for (int i = playback.Count - 1; i >= 0; i--)
            {
                int mask = playback[i].done ? 0 : 1;

                double delta = playback[i].reward +
                               gamma * nextValue * mask - 
                               playback[i].value;

                At = delta + gamma * lambda * mask * At;
                Vt = At + playback[i].value;
                                
                advantages.Insert(0, At);
                returns.Insert(0, Vt);

                nextValue = playback[i].value;
            }
        }
        void ShuffleTrainingData(TrainingData data)
        {
            var rand = new System.Random();
            for (int i = 0; i < data.playback.Count; i++)
            {
                int j = rand.Next(0, data.playback.Count - 1);
                
                // interchange playback memories
                PPOSample temp = data.playback[i];
                data.playback[i] = data.playback[j];
                data.playback[j] = temp;

                // interchange advantage values
                double temp2 = data.advantages[i];
                data.advantages[i] = data.advantages[j];
                data.advantages[j] = temp2;

                // interchange return values
                double temp3 = data.returns[i];
                data.returns[i] = data.returns[j];
                data.returns[j] = temp3;
            }
        }     

    }


    public class TrainingData
    {
        public List<PPOSample> playback = new List<PPOSample>();
        public List<double> advantages = new List<double>();
        public List<double> returns = new List<double>();
    }
}