using NeuroForge;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Serialization;
using UnityEngine;

public class ClassificationDebugger : MonoBehaviour
{
    public NeuralNetwork net;
    public int hiddenUnits = 64;
    public int layerNum = 2;
    public ActivationType activation = ActivationType.Relu;

    [Space]
    [Range(0, 10)] public int whichFuncToLearn = 0;
    [Range(0.1f, 10f)] public double XStddev = 1;
    [Range(0.1f, 10f)] public double YStddev = 1;
    public int batch_size = 100;
    public int epoch = 0;

    [Space]
    public string train_accuracy;
    public string test_accuracy;

    [Space]    
    [Range(0.00001f, 0.1f)] public float learn_rate = 0.1f;
    [Range(0.00000f, 0.1f)] public float regularization = 0.001f;
    [Range(0.00000f, 1f)] public float momentum = 0.9f;

    List<double[]> inputsData = new List<double[]>();
    List<double[]> labelsData = new List<double[]>();

    double[] minsInputs;
    double[] maxsInputs;

    double[] minsLabels;
    double[] maxsLabels;

    List<(double, double, int)> objectsToClassify = new List<(double, double, int)>();

    private void Start()
    {
        net = new NeuralNetwork(2, 2, hiddenUnits, layerNum, activation, ActivationType.Tanh, LossType.MeanSquare, InitializationType.He, true, "classificationTest");
        
        Generate_Data();

        minsInputs = new double[inputsData[0].Length];
        maxsInputs = new double[inputsData[0].Length];

        minsLabels = new double[labelsData[0].Length];
        maxsLabels = new double[labelsData[0].Length];

        FindMinMax();
        Normalize_Inputs_and_Labels();
    }
    private void Update()
    {
        TrainNetwork();
        TestAccuracy();
    }
    double Function(double x)
    {
        switch (whichFuncToLearn)
        {
            case 0:
                return x / (1 + Mathf.Exp((float)-x));
            case 1:
                return (x*x);
            case 2:
                return Math.Cos(x);
            case 3:
                return Math.Sin(x);
            case 4:
                return x / (1 + x * x);
            case 5:
                return x * x * x / Math.Pow(2, x);
            case 6:
                return 0.2 * Math.Pow(x, 4) + 0.1 * Math.Pow(x, 3) - x * x + 1.0;
            case 7:
                return 3 / x * Math.Exp(-0.5 * x);
            case 8:
                return Math.Tanh(x);
            case 9:
                return Math.Sinh(x);
            default:
                return x;
        }
    
    }
    private void Generate_Data()
    {
        for (int i = 0; i < batch_size; i++)
        {
            double x = Functions.RandomGaussian(0, YStddev);
            double y = Functions.RandomGaussian(0, YStddev);
            double[] label = new double[2];
            double funcPoint = Function(x);
            if (y > funcPoint)
            {
                label[0] = 1;
                label[1] = 0;
            }
            else
            {
                label[0] = 0;
                label[1] = 1;
            }

            inputsData.Add(new double[] { x, y });
            labelsData.Add(label);
        }
    }


    private void FindMinMax()
    {
        for (int i = 0; i < minsInputs.Length; i++)
        {
            minsInputs[i] = inputsData.Min(x => x[i]);
            maxsInputs[i] = inputsData.Max(x => x[i]);
        }

        for (int i = 0; i < minsLabels.Length; i++)
        {
            minsLabels[i] = labelsData.Min(x => x[i]);
            maxsLabels[i] = labelsData.Max(x => x[i]);
        }
    }
    private void Normalize_Inputs_and_Labels()
    {
        for (int i = 0; i < inputsData.Count; i++)
        {
            for (int j = 0; j < inputsData[i].Length; j++) // Number of outputs
            {
                // 0 to 1 normalization
                inputsData[i][j] = (inputsData[i][j] - minsInputs[j]) / (maxsInputs[j] - minsInputs[j]);
            }
        }

        for (int i = 0; i < labelsData.Count; i++)
        {
            for (int j = 0; j < labelsData[i].Length; j++) // Number of outputs
            {
                // 0 to 1 normalization
                labelsData[i][j] = (labelsData[i][j] - minsLabels[j]) / (maxsLabels[j] - minsLabels[j]);
            }
        }

    }


    private void TrainNetwork()
    {
        double data_acc = 0;
        for (int i = 0; i < inputsData.Count/2; i++)
        {
            data_acc += net.BackPropagation(inputsData[i], labelsData[i]);
            if (i % 10 == 0) //minibatch
                net.OptimizeParameters(learn_rate, momentum, regularization, true);
        }


        epoch++;

        data_acc /= (inputsData.Count/2);
        data_acc = (1 - data_acc) * 100;
        train_accuracy = data_acc.ToString("0.000") + "%";
    }
    private void TestAccuracy()
    {
        double test_acc = 0;
        objectsToClassify.Clear();
        for (int i = labelsData.Count/2 + 1; i < inputsData.Count; i++)
        {
            double[] inps = inputsData[i];
            double[] outs = net.ForwardPropagation(inps);
            double[] labels = labelsData[i];

            double accuracy_on_sample = 0;
            for (int k = 0; k < outs.Length; k++)
            {
                accuracy_on_sample += Math.Abs((float)(outs[k] - labels[k]));
            }
            accuracy_on_sample /= outs.Length;

            accuracy_on_sample = (1 - accuracy_on_sample) * 100;
            test_acc += accuracy_on_sample;

            int pos = -1;
            if (outs[0] > outs[1])
                pos = 1;
            else
                pos = 0;

            //add to objectsToClassify
            objectsToClassify.Add((inps[0], inps[1], pos));
        }
        test_acc /= (inputsData.Count/2);
        test_accuracy = test_acc.ToString("0.000") + "%";
    }
    private void OnDrawGizmos()
    {
        foreach (var obj in objectsToClassify)
        {
            if (obj.Item3 == 1)
            {
                //Blue Sphere
                Gizmos.color = Color.blue;
                Gizmos.DrawCube(new Vector3((float)obj.Item1, (float)obj.Item2, 0) * 100, Vector3.one * 1f);
            }
            else
            {
                //Red Sphere
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(new Vector3((float)obj.Item1, (float)obj.Item2, 0) * 100, 0.66f);
            }
        }
    }
}