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
using static NeuroForge.Functions;

namespace NeuroForge
{
    [Serializable]
    public class PPOActorNetwork : ScriptableObject
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

        // Gradient Descent
        public void BackPropagation(double[] inputs, double[] losses)
        {
            // losses = Derivative of the loss function values here
            if (weightGradients == null || weightGradients.Length < 1)
                InitGradients();

            NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];

            if (actionSpace == ActionType.Continuous)
            {
                ContinuousForwardPropagation(inputs);
                for (int i = 0; i < outLayer.neurons.Length; i++)
                {
                    if (i % 2 == 0)
                        outLayer.neurons[i].CostValue = losses[i] * Functions.Derivative.TanH(outLayer.neurons[i].InValue);      //mu
                    else
                        outLayer.neurons[i].CostValue = losses[i] * Functions.Derivative.SoftPlus(outLayer.neurons[i].InValue);  //sigma
                }
            } else
            if (actionSpace == ActionType.Discrete)
            {
                double[] rawOuts = DiscreteForwardPropagation(inputs).Item1;

                int rawIndex = 0;
                for (int i = 0; i < outputBranches.Length; i++)
                {
                    double[] rawBranchDerived = new double[outputBranches[i]];
                    Array.Copy(rawOuts, rawIndex, rawBranchDerived, 0, outputBranches[i]);

                    Derivative.SoftMax(rawBranchDerived);

                    for (int j = rawIndex; j < rawIndex + outputBranches[i]; j++)
                    {
                        outLayer.neurons[j].CostValue = losses[j] * rawBranchDerived[j - rawIndex];
                    }

                    rawIndex += outputBranches[i];

                }
            }

            // Calculate hidden layer costs and gradients
            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
                CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1]);
            }

            backwardsCount++;
        }
        public void GradientsClipNorm(float threshold)
        {
            double global_sum = 0;

            // Sum weights' gradients
            foreach (var grad_layer in weightGradients)
            {
                foreach (var clump in grad_layer.weights)
                {
                    foreach (var w_grad in clump)
                    {
                        global_sum += w_grad * w_grad;
                    }
                }
            }

            // Sum biases' gradients
            foreach (var bias_layer in biasGradients)
            {
                foreach (var b_grad in bias_layer.biases)
                {
                    global_sum += b_grad * b_grad;
                }
            }

            double scalar = threshold / Math.Max(threshold, global_sum);

            // Normalize weights
            for (int lay = 0; lay < weightGradients.Length; lay++)
            {
                for (int i = 0; i < weightGradients[lay].weights.Length; i++)
                {
                    for (int j = 0; j < weightGradients[lay].weights[i].Length; j++)
                    {
                        weightGradients[lay].weights[i][j] *= scalar;
                    }
                }
            }

            // Normalize biases
            for (int lay = 0; lay < biasGradients.Length; lay++)
            {
                for (int i = 0; i < biasGradients[lay].biases.Length; i++)
                {
                    biasGradients[lay].biases[i] *= scalar;
                }
            }
        }
        public void OptimiseParameters(float learningRate, float momentum, float regularization)
        {
            learningRate /= backwardsCount;
            backwardsCount = 0;

            double weightDecay = 1 - regularization * learningRate;
            for (int l = 0; l < weightLayers.Length; l++)
            {
                for (int i = 0; i < weightLayers[l].weights.Length; i++)
                {
                    for (int j = 0; j < weightLayers[l].weights[i].Length; j++)
                    {
                        double weight = weightLayers[l].weights[i][j];
                        double veloc = weightMomentums[l].weights[i][j] * momentum - weightGradients[l].weights[i][j] * learningRate;

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
                    double veloc = biasMomentums[i].biases[j] * momentum - biasGradients[i].biases[j] * learningRate;

                    biasMomentums[i].biases[j] = veloc;
                    biasLayers[i].biases[j] += veloc;

                    biasGradients[i].biases[j] = 0;
                }
            }    
        }
       

        private void InitGradients()
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
        private void ActivateLayer(NeuronLayer layer, ActivationType activation)
        {
            if (activation == ActivationType.SoftMax)
            {
                double[] InValuesToActivate = layer.neurons.Select(x => x.InValue).ToArray();
                Activation.SoftMax(InValuesToActivate);
                for (int i = 0; i < InValuesToActivate.Length; i++)
                {
                    layer.neurons[i].OutValue = InValuesToActivate[i];
                }
            }
            else
            {
                foreach (Neuron neuron in layer.neurons)
                {
                    neuron.OutValue = Activation.ActivateValue(neuron.InValue, activation);
                }
            }
        }
        private void CalculateLayerCost(NeuronLayer layer, WeightLayer weights, NeuronLayer nextLayer)
        {
            for (int i = 0; i < layer.neurons.Length; i++)
            {
                double costVal = 0;
                for (int j = 0; j < nextLayer.neurons.Length; j++)
                {
                    costVal += nextLayer.neurons[j].CostValue * weights.weights[i][j];
                }
                costVal *= Derivative.DeriveValue(layer.neurons[i].InValue, activationType);

                layer.neurons[i].CostValue = costVal;
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


        // Continuous
        public PPOActorNetwork(int inputsNum, int continuousSpaceSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initType)
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
                    ActivateLayer(neuronLayers[l], activationType);
                }
            }
            ContinuousActivation();

            double[] outputs = neuronLayers[neuronLayers.Length - 1].GetOutValues();
            float[] continuousActions = GetContinuousActions(outputs);
            return (outputs, continuousActions);
        }
        private void ContinuousActivation()
        {
            NeuronLayer outputLay = neuronLayers[neuronLayers.Length - 1];

            for (int i = 0; i < outputLay.neurons.Length; i++)
            {
                outputLay.neurons[i].OutValue =
                    i % 2 == 0 ?
                    Functions.Activation.TanH(outputLay.neurons[i].InValue) :     // mu
                    Functions.Activation.SoftPlus(outputLay.neurons[i].InValue);  // sigma
            }
        }
        private float[] GetContinuousActions(double[] rawValues)
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
                double sigma = rawContinuousOutputs[i * 2 + 1] + 1e-8;

                double log_prob = -Math.Pow(x - mu, 2) / (2 * sigma * sigma) - Math.Log(Math.Sqrt(2 * Math.PI * sigma * sigma));

                log_probs[i * 2] = log_prob;
                log_probs[i * 2 + 1] = log_prob;
            }
            return log_probs;
        }


        // Discrete
        public PPOActorNetwork(int inputsNum, int[] discreteOutputShape, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initType)
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
                    ActivateLayer(neuronLayers[l], activationType);
                }
            }

            DiscreteActivation();
            double[] outputs = neuronLayers[neuronLayers.Length - 1].GetOutValues();
            int[] discreteActions = GetDiscreteActions(outputs);
            return (outputs, discreteActions);

        }
        private void DiscreteActivation()
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
        private int[] GetDiscreteActions(double[] rawValues)
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
                log_probs[i] = Math.Log(rawDiscreteOutputs[i] + 1e-8);
            }

            return log_probs;
        }

 
        // Other
        private void CreateAsset()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<PPOActorNetwork>("Assets/ActorNN#" + id + ".asset") != null)
                id++;
            string assetName = "ActorNN#" + id + ".asset";

            AssetDatabase.CreateAsset(this, "Assets/" + assetName);
            AssetDatabase.SaveAssets();
            Debug.Log(assetName + " was created!");
        }
        public int GetObservationsNumber() => format[0];
        public int GetActionsNumber() => actionSpace == ActionType.Continuous ? outputBranches[0] : outputBranches.Length; // 1 branch is 1 action for discrete
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
        }//to be deleted
    }
}


