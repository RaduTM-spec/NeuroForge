using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using static NeuroForge.Functions;

namespace NeuroForge
{
    [Serializable]
    public class PPOActor : ScriptableObject
    {
        [SerializeField] public int[] layerFormat;
        [SerializeField] public int[] outputBranches;
        [SerializeField] public NeuronLayer[] neuronLayers;
        [SerializeField] public WeightLayer[] weightLayers;
        [SerializeField] public BiasLayer[] biasLayers;

        [SerializeField] public InitializationType initialization;
        [SerializeField] public ActivationType activationType;
        [SerializeField] public ActionType actionSpace;

        private WeightLayer[] weightGradients;
        private WeightLayer[] weightMomentums;
        private BiasLayer[] biasGradients;
        private BiasLayer[] biasMomentums;


        int backwardsCount = 0;

        // SGD
        private void ZeroGrad()
        {
                biasGradients = new BiasLayer[layerFormat.Length];
                biasMomentums = new BiasLayer[layerFormat.Length];
                weightGradients = new WeightLayer[layerFormat.Length - 1];
                weightMomentums = new WeightLayer[layerFormat.Length - 1];

                for (int i = 0; i < neuronLayers.Length; i++)
                {
                    biasGradients[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);
                    biasMomentums[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);

                }
                for (int i = 0; i < neuronLayers.Length - 1; i++)
                {
                    weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                    weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                }
        }
        public void Backward(double[] inputs, double[] losses)
        {
            if (weightGradients == null || weightGradients.Length < 1)
                ZeroGrad();

            NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];

            if (actionSpace == ActionType.Continuous)
            {
                Forward_Continuous(inputs);
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
                double[] rawOuts = Forward_Discrete(inputs).Item1;

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
        public void GradClipNorm(float threshold)
        {
            double global_sum = 0;

            // Sum weights' gradients square
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

            // Sum biases' gradients square
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
        public void OptimStep(float learningRate, float momentum, float regularization)
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


        // Continuous (mu sigma mu sigma mu sigma)
        const double sigma_scale = 0; // - 1.5, 3.5..  so on
        const double sigma_min = 0.01;
        const double sigma_max = 10.0;
        public PPOActor(int inputsNum, int continuousSpaceSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initType)
        {
            // Set format
            layerFormat = new int[2 + layersNum];
            layerFormat[0] = inputsNum;
            for (int i = 1; i < layerFormat.Length - 1; i++)
            {
                layerFormat[i] = hiddenUnits;
            }
            layerFormat[layerFormat.Length - 1] = continuousSpaceSize * 2;

            // Set types
            this.outputBranches = new int[1];
            this.outputBranches[0] = continuousSpaceSize * 2;
            this.actionSpace = ActionType.Continuous;
            this.initialization = initType;
            this.activationType = activation;

            // Init weight and biases
            neuronLayers = new NeuronLayer[layerFormat.Length];
            biasLayers = new BiasLayer[layerFormat.Length];
            weightLayers = new WeightLayer[layerFormat.Length - 1];
            for (int i = 0; i < neuronLayers.Length; i++)
            {
                neuronLayers[i] = new NeuronLayer(layerFormat[i]);
                biasLayers[i] = new BiasLayer(layerFormat[i], initType);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType);
            }

            CreateAsset();
        }
        public (double[], float[]) Forward_Continuous(double[] inputs)
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
            float[] continuousActions = TransformIntoContinuousActions(outputs);
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
        private float[] TransformIntoContinuousActions(double[] rawValues)
        {
            float[] continuousActions = new float[rawValues.Length/2];
            for (int i = 0; i < rawValues.Length; i += 2)
            {
                double mean = rawValues[i];
                double stddev = rawValues[i + 1];

                // sigma scale
                stddev += sigma_scale;
                stddev = Math.Clamp(stddev, sigma_min, sigma_max);

                double actionSample = Math.Clamp(Functions.RandomGaussian(mean, stddev), -1.0, 1.0);
                continuousActions[i / 2] = (float)actionSample;
            }
            return continuousActions;
        }
        static public double[] GetContinuousLogProbs(double[] rawContinuousOutputs, float[] continuousActions)
        {
            double[] log_probs = new double[rawContinuousOutputs.Length];

            for (int i = 0; i < rawContinuousOutputs.Length; i += 2)
            {
                double x = continuousActions[i / 2];
                double mu = rawContinuousOutputs[i];
                double sigma = rawContinuousOutputs[i + 1]; // +1e-8

                double n = x - mu;
                double f = -(n * n) / (2 * sigma * sigma);
                double log_prob = f - Math.Log(sigma) - Math.Log(Math.Sqrt(2 * Math.PI));

                log_probs[i] = log_prob;        //mu head
                log_probs[i + 1] = log_prob;    //sigma head
            }

            return log_probs;
        }


        // Discrete
        public PPOActor(int inputsNum, int[] discreteOutputShape, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initType)
        {

            // Set format
            layerFormat = new int[2 + layersNum];
            layerFormat[0] = inputsNum;
            for (int i = 1; i < layerFormat.Length - 1; i++)
            {
                layerFormat[i] = hiddenUnits;
            }
            layerFormat[layerFormat.Length - 1] = discreteOutputShape.Sum();

            // Set types
            this.outputBranches = discreteOutputShape;
            this.actionSpace = ActionType.Discrete;
            this.initialization = initType;
            this.activationType = activation;

            // Init weight and biases
            neuronLayers = new NeuronLayer[layerFormat.Length];
            biasLayers = new BiasLayer[layerFormat.Length];
            weightLayers = new WeightLayer[layerFormat.Length - 1];
            for (int i = 0; i < neuronLayers.Length; i++)
            {
                neuronLayers[i] = new NeuronLayer(layerFormat[i]);
                biasLayers[i] = new BiasLayer(layerFormat[i], initType);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType);
            }
            CreateAsset();            
        }
        public (double[], int[]) Forward_Discrete(double[] inputs)
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
            int[] discreteActions = TransformIntoDiscreteActions(outputs);
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
        private int[] TransformIntoDiscreteActions(double[] rawValues)
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

                int discreteAction = Functions.ArgMax(BRANCH_VALUES);
                discreteActions[br] = discreteAction;
            }

            return discreteActions;
        }
        static public double[] GetDiscreteLogProbs(double[] rawDiscreteOutputs)
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
            while (AssetDatabase.LoadAssetAtPath<PPOActor>("Assets/Actor#" + id + ".asset") != null)
                id++;
            string assetName = "Actor#" + id + ".asset";

            AssetDatabase.CreateAsset(this, "Assets/" + assetName);
            AssetDatabase.SaveAssets();
        }
        public int GetNoObservations() => layerFormat[0];
        public int GetNoParallelActions() => actionSpace == ActionType.Continuous ? outputBranches[0] : outputBranches.Length; // 1 branch is 1 action for discrete
    }
}


