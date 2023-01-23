using NeuroForge;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;
using UnityEngine.Windows;

namespace NeuroForge
{
    [Serializable]
    public class ActorNetwork : ScriptableObject
    {
        [SerializeField] public int[] format;
        [SerializeField] public int[] outputBranches;
        [SerializeField] public NeuronLayer[] neuronLayers;
        [SerializeField] public WeightLayer[] weightLayers;
        [SerializeField] public BiasLayer[] biasLayers;

        [SerializeField] public ActivationType activationType;
        [SerializeField] public ActionType actionSpace;

        private WeightLayer[] weightGradients;
        private WeightLayer[] weightMomentums;
        private BiasLayer[] biasGradients;
        private BiasLayer[] biasMomentums;

        int backwardsCount = 0;

        #region Gradient Descent
        public void BackPropagation(double[] inputs, double[] losses)
        {
            NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];
            if (actionSpace == ActionType.Continuous)
            {
                ContinuousForwardPropagation(inputs);
                for (int i = 0; i < outLayer.neurons.Length; i++)
                {
                    if (i % 2 == 0)
                        outLayer.neurons[i].CostValue = losses[i] * Functions.Derivative.DerivativeTanh(outLayer.neurons[i].InValue);      //mu
                    else
                        outLayer.neurons[i].CostValue = losses[i] * Functions.Derivative.DerivativeSoftPlus(outLayer.neurons[i].InValue);  //sigma
                }
            } else
            if (actionSpace == ActionType.Discrete)
            {
                double[] rawOuts = DiscreteForwardPropagation(inputs).Item1;

                int rawIndex = 0;
                for (int i = 0; i < outputBranches.Length; i++)
                {
                    double[] rawBranchToDerive = new double[outputBranches[i]];
                    Array.Copy(rawOuts, rawIndex, rawBranchToDerive, 0, outputBranches[i]);

                    Functions.Derivative.DerivativeSoftMax(rawBranchToDerive);

                    for (int j = rawIndex; j < rawIndex + outputBranches[i]; j++)
                    {
                        outLayer.neurons[j].CostValue = losses[j] * rawBranchToDerive[j - rawIndex];
                    }

                    rawIndex += outputBranches[i];

                }
            }

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                Functions.Cost.CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1], activationType);
            }
            backwardsCount++;
        }
        public void OptimizeParameters(float learningRate, float momentum, float regularization, bool descent)
        {
            ApplyGradients(learningRate / backwardsCount, momentum, regularization, descent ? -1 : 1);
            backwardsCount = 0;
        }
        public void ZeroGradients()
        {
            if(weightGradients == null || weightGradients.Length == 0)
            {
                biasGradients = new BiasLayer[format.Length];
                biasMomentums = new BiasLayer[format.Length];
                weightGradients = new WeightLayer[format.Length - 1];
                weightMomentums = new WeightLayer[format.Length - 1];

                for (int i = 0; i < neuronLayers.Length; i++)
                {
                    biasGradients[i] = new BiasLayer(format[i], InitializationType.Zero);
                    biasMomentums[i] = new BiasLayer(format[i], InitializationType.Zero);

                }
                for (int i = 0; i < neuronLayers.Length - 1; i++)
                {
                    weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                    weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                }
            }
            else
            {
                for (int i = 0; i < neuronLayers.Length; i++)
                {
                    biasLayers[i].Zero();
                    biasMomentums[i].Zero();
                }
                for (int i = 0; i < neuronLayers.Length - 1; i++)
                {
                    weightGradients[i].Zero();
                    weightMomentums[i].Zero();
                }
            }
            
        }
        private void UpdateGradients(WeightLayer weightGradient, BiasLayer biasGradient, NeuronLayer previousNeuronLayer, NeuronLayer nextNeuronLayer)
        {
            //Related to Backpropagation
            lock (weightGradient)
            {
                for (int i = 0; i < previousNeuronLayer.neurons.Length; i++)
                {

                    for (int j = 0; j < nextNeuronLayer.neurons.Length; j++)
                    {
                        weightGradient.weights[i][j] += previousNeuronLayer.neurons[i].OutValue * nextNeuronLayer.neurons[j].CostValue;
                    }
                }
            }
            lock (biasGradient)
            {
                for (int i = 0; i < nextNeuronLayer.neurons.Length; i++)
                {
                    biasGradient.biases[i] += 1 * nextNeuronLayer.neurons[i].CostValue;
                }
            }
        }
        private void ApplyGradients(float modifiedLearnRate, float momentum, float regularization, double direction)
        {
            //Related to UpdateParameters
            double weightDecay = 1 - regularization * modifiedLearnRate;
            for (int l = 0; l < weightLayers.Length; l++)
            {
                for (int i = 0; i < weightLayers[l].weights.Length; i++)
                {
                    for (int j = 0; j < weightLayers[l].weights[i].Length; j++)
                    {
                        double weight = weightLayers[l].weights[i][j];
                        double veloc = weightMomentums[l].weights[i][j] * momentum + weightGradients[l].weights[i][j] * modifiedLearnRate * direction;

                        weightMomentums[l].weights[i][j] = veloc;
                        weightLayers[l].weights[i][j] = weight * weightDecay + veloc;

                        //Reset the gradient
                        weightGradients[l].weights[i][j] = 0;
                    }
                }
            }
            for (int i = 0; i < biasLayers.Length; i++)
            {
                for (int j = 0; j < biasLayers[i].biases.Length; j++)
                {
                    double bias = biasLayers[i].biases[j];
                    double veloc = biasMomentums[i].biases[j] * momentum + biasGradients[i].biases[j] * modifiedLearnRate * direction;

                    biasMomentums[i].biases[j] = veloc;
                    biasLayers[i].biases[j] += veloc;

                    biasGradients[i].biases[j] = 0;
                }
            }
        }
       

        #endregion

        #region Continuous
        public ActorNetwork(int inputsNum, int continuousSpaceSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initType)
        {
            // Set format
            format = new int[2 + layersNum];
            format[0] = inputsNum;
            for (int i = 1; i < format.Length - 1; i++)
            {
                format[i] = hiddenUnits;
            }
            format[format.Length - 1] = continuousSpaceSize * 2;

            // Set types
            this.outputBranches = new int[1];
            this.outputBranches[0] = continuousSpaceSize * 2;
            this.actionSpace = ActionType.Continuous;
            this.activationType = activation;

            // Init weight and biases
            neuronLayers = new NeuronLayer[format.Length];
            biasLayers = new BiasLayer[format.Length];
            weightLayers = new WeightLayer[format.Length - 1];
            for (int i = 0; i < neuronLayers.Length; i++)
            {
                neuronLayers[i] = new NeuronLayer(format[i]);
                biasLayers[i] = new BiasLayer(format[i], initType);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType);
            }

            CreateAsset();
        }
        public (double[], float[]) ContinuousForwardPropagation(double[] inputs)
        {
            if (actionSpace != ActionType.Continuous)
                throw new Exception("Action space for this model was set to Discrete");

            neuronLayers[0].SetOutValues(inputs);
            for (int l = 1; l < neuronLayers.Length; l++)
            {
                for (int n = 0; n < neuronLayers[l].neurons.Length; n++)
                {
                    double sumValue = biasLayers[l].biases[n];
                    for (int prevn = 0; prevn < neuronLayers[l - 1].neurons.Length; prevn++)
                    {
                        sumValue += neuronLayers[l - 1].neurons[prevn].OutValue * weightLayers[l - 1].weights[prevn][n];
                    }
                    neuronLayers[l].neurons[n].InValue = sumValue;
                }

                //Activate neuron layer
                if (l < neuronLayers.Length - 1)
                {
                    Functions.Activation.ActivateLayer(neuronLayers[l], activationType);
                }
            }
            ActivateRawContinuous();

            double[] outputs = neuronLayers[neuronLayers.Length - 1].GetOutValues();
            float[] continuousActions = FromRawToContinuous(outputs);
            return (outputs, continuousActions);
        }
        private void ActivateRawContinuous()
        {
            NeuronLayer outputLay = neuronLayers[neuronLayers.Length - 1];

            for (int i = 0; i < outputLay.neurons.Length; i++)
            {
                outputLay.neurons[i].OutValue =
                    i % 2 == 0 ?
                    Functions.Activation.Tanh(outputLay.neurons[i].InValue) :     // mu
                    Functions.Activation.SoftPlus(outputLay.neurons[i].InValue);  // sigma
            }
        }
        private float[] FromRawToContinuous(double[] rawValues)
        {

            float[] continuousActions = new float[rawValues.Length];
            for (int i = 0; i < rawValues.Length; i += 2)
            {
                double mean = rawValues[i];
                double stddev = rawValues[i + 1];
                if (stddev == 0) stddev = 1e-8;
                double actionSample = Math.Clamp(Functions.RandomGaussian(mean, stddev), -1.0, 1.0);

                continuousActions[i / 2] = (float)actionSample;
            }
            return continuousActions;
        }
        public double[] GetContinuousLogProbs(double[] rawContinuousOutputs, float[] continuousActions)
        {
            double[] log_probs = new double[rawContinuousOutputs.Length];

            for (int i = 0; i < continuousActions.Length/2; i++)
            {
                double x = continuousActions[i];
                double mu = rawContinuousOutputs[i * 2];
                double sigma = rawContinuousOutputs[i * 2 + 1];
                if (sigma == 0) sigma = +1e-8;
                double log_prob = -Math.Pow(x - mu, 2) / (2 * sigma * sigma) - Math.Log(2 * Math.PI * sigma * sigma);
                if (log_prob == double.NaN)
                    log_prob = 1e-8;
                log_probs[i * 2] = log_prob;
                log_probs[i * 2 + 1] = log_prob;
            }
            return log_probs;
        }

        #endregion

        #region Discrete
        public ActorNetwork(int inputsNum, int[] discreteOutputShape, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initType)
        {

            // Set format
            format = new int[2 + layersNum];
            format[0] = inputsNum;
            for (int i = 1; i < format.Length - 1; i++)
            {
                format[i] = hiddenUnits;
            }
            format[format.Length - 1] = discreteOutputShape.Sum();

            // Set types
            this.outputBranches = discreteOutputShape;
            this.actionSpace = ActionType.Discrete;
            this.activationType = activation;

            // Init weight and biases
            neuronLayers = new NeuronLayer[format.Length];
            biasLayers = new BiasLayer[format.Length];
            weightLayers = new WeightLayer[format.Length - 1];
            for (int i = 0; i < neuronLayers.Length; i++)
            {
                neuronLayers[i] = new NeuronLayer(format[i]);
                biasLayers[i] = new BiasLayer(format[i], initType);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType);
            }
            CreateAsset();            
        }
        public (double[], int[]) DiscreteForwardPropagation(double[] inputs)
        {
            if (actionSpace != ActionType.Discrete)
                throw new Exception("Action space for this model was set to Continuous");

            neuronLayers[0].SetOutValues(inputs);
            for (int l = 1; l < neuronLayers.Length; l++)
            {
                for (int n = 0; n < neuronLayers[l].neurons.Length; n++)
                {
                    double sumValue = biasLayers[l].biases[n];
                    for (int prevn = 0; prevn < neuronLayers[l - 1].neurons.Length; prevn++)
                    {
                        sumValue += neuronLayers[l - 1].neurons[prevn].OutValue * weightLayers[l - 1].weights[prevn][n];
                    }
                    neuronLayers[l].neurons[n].InValue = sumValue;
                }

                //Activate neuron layer
                if (l < neuronLayers.Length - 1)
                {
                    Functions.Activation.ActivateLayer(neuronLayers[l], activationType);
                }
            }

            ActivateRawDiscrete();
            double[] outputs = neuronLayers[neuronLayers.Length - 1].GetOutValues();
            int[] discreteActions = FromRawToDiscrete(outputs);
            return (outputs, discreteActions);

        }
        private void ActivateRawDiscrete()
        {
            NeuronLayer outputLay = neuronLayers[neuronLayers.Length - 1];

            List<double> rawValues = outputLay.neurons.Select(x => x.InValue).ToList();
            int index = 0;

            // Foreach branch, activate the branch values
            foreach (var branch in outputBranches)
            {
                // Get the branch from raw values
                double[] branchValues = rawValues.GetRange(index, branch).ToArray();

                // Activate the branch
                Functions.Activation.SoftMax(branchValues);

                // Place the activated branch on OutValues
                for (int i = index; i < index + branch; i++)
                {
                    outputLay.neurons[i].OutValue = branchValues[i - index];
                }

                index += branch;
            }
        }
        private int[] FromRawToDiscrete(double[] rawValues)
        {
            int[] discreteActions = new int[outputBranches.Length];

            int indexInRawOutputs = 0;
            for (int br = 0; br < outputBranches.Length; br++)
            {
                double[] BRANCH_VALUES = new double[outputBranches[br]];
                for (int i = 0; i < BRANCH_VALUES.Length; i++)
                {
                    BRANCH_VALUES[i] = rawValues[indexInRawOutputs++];
                }

                int discreteAction = Functions.Activation.ArgMax(BRANCH_VALUES);
                discreteActions[br] = discreteAction;
            }

            return discreteActions;
        }
        public static double[] GetDiscreteLogProbs(double[] rawDiscreteOutputs)
        {
            double[] log_probs = new double[rawDiscreteOutputs.Length];

            for (int i = 0; i < rawDiscreteOutputs.Length; i++)
            {
                log_probs[i] = Math.Log(rawDiscreteOutputs[i]);
                if (log_probs[i] == double.NaN)
                    log_probs[i] = 1e-8;
            }

            return log_probs;
        }

        #endregion

        #region Other
        private void CreateAsset()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<ActorNetwork>("Assets/ActorNN#" + id + ".asset") != null)
                id++;
            string assetName = "ActorNN#" + id + ".asset";

            AssetDatabase.CreateAsset(this, "Assets/" + assetName);
            AssetDatabase.SaveAssets();
            Debug.Log(assetName + " was created!");
        }
        public int GetObservationsNumber() => format[0];
        public int GetActionsNumber() => actionSpace == ActionType.Continuous ? outputBranches[0] : outputBranches.Length;
        public double GetMaxGradientValue()
        {
            double max = 0;
            for (int i = 0; i < weightGradients.Length; i++)
            {
                for (int j = 0; j < weightGradients[i].weights.Length; j++)
                {
                    for (int k = 0; k < weightGradients[i].weights[j].Length; k++)
                    {
                        if (weightGradients[i].weights[j][k] > max)
                        {
                            max = weightGradients[i].weights[j][k];
                        }
                    }
                }
            }
            return max;
        }

        #endregion
    }
}


