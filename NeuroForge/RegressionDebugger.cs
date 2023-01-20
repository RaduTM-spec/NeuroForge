using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
using UnityEditor;
using Unity.VisualScripting;
using System.Net.Sockets;
using System.Linq;
using UnityEngine.UIElements;
using System.Text;
using System;
using Newtonsoft.Json.Linq;

public class RegressionDebugger : MonoBehaviour
{
    public NeuralNetwork net;
    public int hiddenUnits = 64;
    public int layerNum = 2;
    public InitializationType initType = InitializationType.He;
    public ActivationType activation = ActivationType.Relu;

    [Space]
    [Range(0, 10)] public int whichFuncToLearn = 0;
    [Range(0.1f,10f)]public double dataStddev = 1;
    public int batch_size = 100;
    public int epoch = 0;

    [Space]
    public string train_accuracy;
    public string test_accuracy; public Color test_color = Color.red;

    [Space]
    [Range(0.00001f,1f)] public float learn_rate = 0.1f;
    [Range(0.00000f, 1f)] public float momentum = 0.9f;
    [Range(0.00000f, 0.1f)] public float regularization = 0.001f;
  

    List<double[]> inputsData = new List<double[]>();
    List<double[]> labelsData = new List<double[]>();

    double[] minsInputs;
    double[] maxsInputs;

    double[] minsLabels;
    double[] maxsLabels;

    List<(double, double)> targetDots = new List<(double, double)>();
    List<(double, double)> testDots = new List<(double, double)>();

    private void Start()
    {
        net = new NeuralNetwork(1, 1, hiddenUnits, layerNum, activation, ActivationType.Tanh, LossType.MeanSquare, initType,true, "regressionTest");
        Generate_Data();

        minsInputs = new double[inputsData[0].Length];
        maxsInputs = new double[inputsData[0].Length];

        minsLabels = new double[labelsData[0].Length];
        maxsLabels = new double[labelsData[0].Length];

        Normalize_Inputs_and_Labels();
        for (int i = 0; i < inputsData.Count; i++)
        {
            targetDots.Add((inputsData[i][0], labelsData[i][0]));
        }
    }
    private void Update()
    {
        TrainNetwork();
        TestAccuracy();
    }

    private double Function(double x)
    {
        switch (whichFuncToLearn)
        {
            case 0:
                return x / (1 + Mathf.Exp((float)-x));
            case 1:
                return (x * x);
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
            double input = Functions.RandomGaussian(0, dataStddev);
            double label = Function(input);
            inputsData.Add(new double[] { input });
            labelsData.Add(new double[] { label });
        }     
    }

    private void Normalize_Inputs_and_Labels()
    {

        // Find Min Max
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
             double[] outs = net.ForwardPropagation(inputsData[i]);
             data_acc += net.BackPropagation(inputsData[i], labelsData[i]);

             if(i%10 == 0)
                net.OptimizeParameters(learn_rate, momentum, regularization, true);
        }
        
        epoch++;

        data_acc /= inputsData.Count;
        data_acc = (1 - data_acc) * 100;
        train_accuracy = data_acc.ToString("0.000") + "%";
    }
    private void TestAccuracy()
    {    
        double test_acc = 0;
        testDots.Clear();
        for (int i = inputsData.Count/2+1; i < inputsData.Count; i++)
        {
            double[] inps = inputsData[i];
            double[] outs = net.ForwardPropagation(inps);
            double[] labels = inputsData[i];

            testDots.Add((inps[0], outs[0]));

            double accuracy_on_sample = Mathf.Abs((float)(outs[0] - labels[0]));
            
            accuracy_on_sample = (1 - accuracy_on_sample) * 100;
            test_acc += accuracy_on_sample;
        }
        test_acc /= inputsData.Count/2;
        test_accuracy = test_acc.ToString("0.000") + "%";
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;//draw real
        foreach (var dot in targetDots)
        {
            Vector3 pos = new Vector3((float)dot.Item1, (float)dot.Item2, 0) * 100;
            Gizmos.DrawSphere(pos, 1f);
        }

        Gizmos.color = test_color;
        foreach (var dot in testDots)
        {
            Vector3 pos = new Vector3((float)dot.Item1, (float)dot.Item2, 0) * 100;
            Gizmos.DrawSphere(pos, 1f);
        }
        
    }
}
