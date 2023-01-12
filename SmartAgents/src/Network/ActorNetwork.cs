using SmartAgents;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;
using UnityEngine.Windows;


public class ActorNetwork : ScriptableObject
{
    [SerializeField] public int[] format;
    [SerializeField] public int[] outputShape;
    [SerializeField] public NeuronLayer[] neuronLayers;
    [SerializeField] public WeightLayer[] weightLayers;
    [SerializeField] public BiasLayer[] biasLayers;

    [SerializeField] public ActivationType activationType;
    [SerializeField] public ActionType actionSpace;

    private WeightLayer[] weightGradients;
    private WeightLayer[] weightMomentums;
    private BiasLayer[] biasGradients;
    private BiasLayer[] biasMomentums;

    int backPropagationsCount = 0;

    // Gradient Descent
    public void BackPropagation(double[] inputs, double[] losses)
    {
        if (weightGradients == null || weightGradients.Length == 0)
            InitGradients_InitMomentums();
        
        NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];
        if(actionSpace == ActionType.Continuous)
        {
            for (int i = 0; i < outLayer.neurons.Length; i++)
            {
                if (i % 2 == 0)
                    outLayer.neurons[i].CostValue = losses[i] * Functions.Derivative.DerivativeTanh(outLayer.neurons[i].InValue);      //mu
                else                                          
                    outLayer.neurons[i].CostValue = losses[i] * Functions.Derivative.DerivativeSoftPlus(outLayer.neurons[i].InValue);  //sigma
            }
        }else
        if(actionSpace == ActionType.Discrete)
        {
            double[] rawOuts = DiscreteForwardPropagation(inputs).Item1;
            
            int rawIndex = 0;
            for (int i = 0; i < outputShape.Length; i++)
            {
                double[] rawBranchToDerive = new double[outputShape[i]];
                Array.Copy(rawOuts, rawIndex, rawBranchToDerive, 0, outputShape[i]);

                Functions.Derivative.DerivativeSoftMax(rawBranchToDerive);

                for (int j = rawIndex; j < rawIndex + outputShape[i]; j++)
                {
                    outLayer.neurons[j].CostValue = losses[j] * rawBranchToDerive[j - rawIndex];
                }

                rawIndex += outputShape[i];
                
            }
        }

        for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
        {
            UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], neuronLayers[wLayer], neuronLayers[wLayer + 1]);
            Functions.Cost.CalculateLayerCost(neuronLayers[wLayer], weightLayers[wLayer], neuronLayers[wLayer + 1], activationType);
        }
        backPropagationsCount++;
    }
    public void OptimizeParameters(float learningRate, float momentum, float regularization, bool descent)
    {
        ApplyGradients(learningRate / backPropagationsCount, momentum, regularization, descent?-1:1);
        backPropagationsCount = 0;
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
    private void InitGradients_InitMomentums()
    {
        biasGradients = new BiasLayer[format.Length];
        biasMomentums = new BiasLayer[format.Length];
        weightGradients = new WeightLayer[format.Length - 1];
        weightMomentums = new WeightLayer[format.Length - 1];

        for (int i = 0; i < neuronLayers.Length; i++)
        {
            biasGradients[i] = new BiasLayer(format[i], true);
            biasMomentums[i] = new BiasLayer(format[i], true);

        }
        for (int i = 0; i < neuronLayers.Length - 1; i++)
        {
            weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1, true);
            weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1, true);
        }


    }


    // Continuous section
    public ActorNetwork(int inputsNum, int continuousSpaceSize, int hiddenUnits, int layersNum, ActivationType activation)
    {
        // Set format
        format = new int[2 + layersNum];
        format[0] = inputsNum;
        for (int i = 0; i < format.Length - 2; i++)
        {
            format[i] = hiddenUnits;
        }
        format[format.Length - 1] = continuousSpaceSize * 2;

        // Set types
        this.outputShape = new int[1];
        this.outputShape[0] = continuousSpaceSize * 2;
        this.actionSpace = ActionType.Continuous;
        this.activationType = activation;

        // Init weight and biases
        neuronLayers = new NeuronLayer[format.Length];
        biasLayers = new BiasLayer[format.Length];
        weightLayers = new WeightLayer[format.Length - 1];
        for (int i = 0; i < neuronLayers.Length; i++)
        {
            neuronLayers[i] = new NeuronLayer(format[i]);
            biasLayers[i] = new BiasLayer(format[i]);

        }
        for (int i = 0; i < neuronLayers.Length - 1; i++)
        {
            weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1);
        }

        // Create asset
        string name = GenerateActorName();
        Debug.Log(name + " was created!");
        AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
        AssetDatabase.SaveAssets();
    }
    public (double[],float[]) ContinuousForwardPropagation(double[] inputs)
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
        return (outputs,continuousActions);
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
            double stddev = rawValues[i + 1] + 0.00000001;

            double actionSample = Math.Clamp(Functions.RandomGaussian(mean, stddev), -1.0, 1.0);

            continuousActions[i / 2] = (float)actionSample;
        }
        return continuousActions;
    }
    public double[] GetContinuousLogProbs(double[] rawContinuousOutputs, float[] continuousActions)
    {
        Debug.LogError("NOT IMPLEMENTED");
        return null;
    }


    // Discrete section
    public ActorNetwork(int inputsNum, int[] discreteOutputShape, int hiddenUnits, int layersNum, ActivationType activation)
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
        this.outputShape = discreteOutputShape;
        this.actionSpace = ActionType.Discrete;
        this.activationType = activation;

        // Init weight and biases
        neuronLayers = new NeuronLayer[format.Length];
        biasLayers = new BiasLayer[format.Length];
        weightLayers = new WeightLayer[format.Length - 1];
        for (int i = 0; i < neuronLayers.Length; i++)
        {
            neuronLayers[i] = new NeuronLayer(format[i]);
            biasLayers[i] = new BiasLayer(format[i]);

        }
        for (int i = 0; i < neuronLayers.Length - 1; i++)
        {
            weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], 1);
        }

        // Create asset
        string name = GenerateActorName();
        
        AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
        AssetDatabase.SaveAssets();
        Debug.Log(name + " was created!");
    }
    public (double[],int[]) DiscreteForwardPropagation(double[] inputs)
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
        return (outputs,discreteActions);

    }
    private void ActivateRawDiscrete()
    {
        NeuronLayer outputLay = neuronLayers[neuronLayers.Length - 1];

        List<double> rawValues = outputLay.neurons.Select(x => x.InValue).ToList();
        int index = 0;

        // Foreach branch, activate the branch values
        foreach (var branch in outputShape)
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
        int[] discreteActions = new int[outputShape.Length];

        int indexInRawOutputs = 0;
        for (int br = 0; br < outputShape.Length; br++)
        {
            double[] BRANCH_VALUES = new double[outputShape[br]];
            for (int i = 0; i < BRANCH_VALUES.Length; i++)
            {
                BRANCH_VALUES[i] = rawValues[indexInRawOutputs++];
            }

            int discreteAction = DecideDiscreteBranchAction(BRANCH_VALUES);
            discreteActions[br] = discreteAction;
        }

        return discreteActions;
    }
    private int DecideDiscreteBranchAction(double[] rawBranchOutputs)
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
    public static double[] GetDiscreteLogProbs(double[] rawDiscreteOutputs)
    {
        double[] log_probs = new double[rawDiscreteOutputs.Length];

        for (int i = 0; i < rawDiscreteOutputs.Length; i++)
        {
            log_probs[i] = Math.Log(rawDiscreteOutputs[i]);
        }
        return log_probs;
    }


    // Other
    private string GenerateActorName()
    {
        short id = 1;
        while (AssetDatabase.LoadAssetAtPath<NeuralNetwork>("Assets/ActorNN#" + id + ".asset") != null)
            id++;
        return "ActorNN#" + id;
    }
    public int GetObservationsNumber() => format[0];
    public int GetActionsNumber() => actionSpace == ActionType.Continuous? outputShape[0]: outputShape.Length;

}
