using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using static NeuroForge.Functions;

public class ReccurentNeuralNetwork : ScriptableObject
{
    [SerializeField] public int[] layerFormat;
    [SerializeField] public NeuronLayer[] neuronLayers;
    [SerializeField] public WeightLayer[] weightLayers;
    [SerializeField] public BiasLayer[] biasLayers;

    [SerializeField] public InitializationType initializationType; // shows how the network was initialized
    [SerializeField] public ActivationType activationType;
    [SerializeField] public ActivationType outputActivationType;
    [SerializeField] public LossType lossType;

    private WeightLayer[] weightGradients;
    private WeightLayer[] weightMomentums;
    private BiasLayer[] biasGradients;
    private BiasLayer[] biasMomentums;

    private double[][] sequencialWeightsGradients;
    private double[][] sequencialWeightsMomentums;

    public ReccurentNeuralNetwork(int inputs, int outputs, int hiddenUnits, int hiddenLayersNumber,
                            ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction,
                            InitializationType initType, bool createAsset, string name)
    {
        this.layerFormat = GetFormat(inputs, outputs, hiddenUnits, hiddenLayersNumber);
        this.initializationType = initType;
        this.activationType = activationFunction;
        this.outputActivationType = outputActivationFunction;
        this.lossType = lossFunction;

        neuronLayers = new NeuronLayer[layerFormat.Length];
        biasLayers = new BiasLayer[layerFormat.Length];
        weightLayers = new WeightLayer[layerFormat.Length - 1];


        for (int i = 0; i < neuronLayers.Length; i++)
        {
            if (i == 0)
                neuronLayers[i] = new NeuronLayer(layerFormat[i],activationType);
            else if (i == neuronLayers.Length - 1)
                neuronLayers[i] = new NeuronLayer(layerFormat[i], outputActivationType);
            else
                neuronLayers[i] = new SequencialNeuronLayer(layerFormat[i], activationType, initType);

            biasLayers[i] = new BiasLayer(layerFormat[i], initType);

        }
        for (int i = 0; i < neuronLayers.Length - 1; i++)
        {
            weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType);
        }

        if (createAsset)
        {
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
        }
    }

    private double[] Forward(double[] inputs)
    {
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
                neuronLayers[l].neurons[n].InValue += sumValue;
            }

            neuronLayers[l].Activate();
        }
        var outputs = neuronLayers[neuronLayers.Length - 1].GetOutValues();

        // After one forward pass we need to reset each neuron layer
        foreach (var nl in neuronLayers)
        {
            nl.Reset();
        }

        return outputs;
    }
    public List<double[]> Forward(List<double[]> stacked_inputs)
    {
        List<double[]> sequence_outputs = new List<double[]>();
        foreach (var inp in stacked_inputs)
        {
            sequence_outputs.Add(Forward(inp));
        }
        return sequence_outputs;
    }
    
    public double Backward(List<double[]> stacked_inputs, double[] labels)
    {
        if (weightGradients == null || weightGradients.Length < 1)
            ZeroGrad();

        Forward(stacked_inputs);
        double error = CalculateOutputLayerCost(labels);

        // here is the final dance
        for(int wLayer = weightGradients.Length - 1; wLayer >= 0; wLayer--)
        {
            // we need to do some unroll and stuff... quite complicated for now
        }

        return error;
    }

    private double CalculateOutputLayerCost(double[] labels)
    {
        NeuronLayer outLayer = neuronLayers[neuronLayers.Length - 1];
        double err = 0;
        if (outputActivationType != ActivationType.SoftMax)
        {
            for (int i = 0; i < outLayer.neurons.Length; i++)
            {
                switch (lossType)
                {
                    case LossType.MeanSquare:
                        outLayer.neurons[i].CostValue = Loss.MeanSquareDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                        err += Error.MeanSquare(outLayer.neurons[i].OutValue, labels[i]);
                        break;
                    case LossType.CrossEntropy:
                        outLayer.neurons[i].CostValue = Loss.CrossEntropyDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                        err += Error.CrossEntropy(outLayer.neurons[i].OutValue, labels[i]);
                        break;
                    case LossType.MeanAbsolute:
                        outLayer.neurons[i].CostValue = Loss.AbsoluteDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                        err += Error.MeanAbsolute(outLayer.neurons[i].OutValue, labels[i]);
                        break;
                }
            }
        }
        else
        {
            double[] derivedInValuesBySoftMax = new double[labels.Length];
            for (int i = 0; i < derivedInValuesBySoftMax.Length; i++)
                derivedInValuesBySoftMax[i] = outLayer.neurons[i].InValue;

            Derivative.SoftMax(derivedInValuesBySoftMax);

            for (int i = 0; i < outLayer.neurons.Length; i++)
            {
                switch (lossType)
                {
                    case LossType.MeanSquare:
                        outLayer.neurons[i].CostValue = Loss.MeanSquareDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                        err += Error.MeanSquare(outLayer.neurons[i].OutValue, labels[i]);
                        break;
                    case LossType.CrossEntropy:
                        outLayer.neurons[i].CostValue = Loss.CrossEntropyDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                        err += Error.CrossEntropy(outLayer.neurons[i].OutValue, labels[i]);
                        break;
                    case LossType.MeanAbsolute:
                        outLayer.neurons[i].CostValue = Loss.AbsoluteDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                        err += Error.MeanAbsolute(outLayer.neurons[i].OutValue, labels[i]);
                        break;
                }
            }



        }

        return err / labels.Length;
    }



    private void ZeroGrad()
    {
        biasGradients = new BiasLayer[layerFormat.Length];
        biasMomentums = new BiasLayer[layerFormat.Length];
        weightGradients = new WeightLayer[layerFormat.Length - 1];
        weightMomentums = new WeightLayer[layerFormat.Length - 1];
        sequencialWeightsGradients = new double[neuronLayers.Length - 2][];
        sequencialWeightsMomentums = new double[neuronLayers.Length - 2][];

        for (int i = 0; i < neuronLayers.Length; i++)
        {
            biasGradients[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);
            biasMomentums[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);

        }
        for (int i = 0; i < weightLayers.Length; i++)
        {
            weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
            weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
        }
        for (int i = 0; i < neuronLayers.Length - 2; i++)
        {
            sequencialWeightsGradients[i - 1] = new double[neuronLayers[i + 1].neurons.Length];
            sequencialWeightsMomentums[i - 1] = new double[neuronLayers[i + 1].neurons.Length];
        }
    }


    static int[] GetFormat(int inputs, int outs, int hidden_units, int hidden_lay_num)
    {
        int[] form = new int[2 + hidden_lay_num];

        form[0] = inputs;
        for (int i = 1; i <= hidden_lay_num; i++)
        {
            form[i] = hidden_units;
        }
        form[form.Length - 1] = outs;

        return form;
    }
}
