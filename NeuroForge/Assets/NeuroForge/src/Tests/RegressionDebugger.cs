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

public class RegressionDebugger : MonoBehaviour
{
    public NeuralNetwork net;
    public ComputeShader computeShader;
    public int hiddenUnits = 64;
    public int layerNum = 2;
    public InitializationType initType = InitializationType.He;
    public ActivationType activation = ActivationType.Relu;
    public ActivationType outActivation = ActivationType.Linear;
    public NormlizationType normalization = NormlizationType.MinusOneOne;

    [Space]
    [Range(0, 7)] public int whichFuncToLearn = 0;
    [Range(0.1f,10f)]public double dataStddev = 1;
    public int data_size = 100;
    public int batch_size = 10;
    public int epoch = 0;

    [Space]
    public string train_accuracy;
    public string test_accuracy; public Color test_color = Color.red;

    [Space]
    [Range(0.00001f, 0.1f)] public float learn_rate = 0.1f;
    [Range(0.00000f, 1f)] public float momentum = 0.9f;
    [Range(0.00000f, 0.1f)] public float regularization = 0.001f;
  

    List<double[]> inputsData = new List<double[]>();
    List<double[]> labelsData = new List<double[]>();

    RunningNormalizer inputsNormalizer = new RunningNormalizer(1);
    RunningNormalizer labelsNormalizer = new RunningNormalizer(1);

    double[] minsInputs;
    double[] maxsInputs;

    double[] minsLabels;
    double[] maxsLabels;

    List<(double, double)> targetDots = new List<(double, double)>();
    List<(double, double)> testDots = new List<(double, double)>();


    private void Start()
    {
        net = new NeuralNetwork(1, 1, hiddenUnits, layerNum, activation, outActivation, LossType.MeanSquare, initType, true, "regressionTest");
        Generate_Data();

        minsInputs = new double[inputsData[0].Length];
        maxsInputs = new double[inputsData[0].Length];

        minsLabels = new double[labelsData[0].Length];
        maxsLabels = new double[labelsData[0].Length];

        Normalize_Inputs_and_Labels();

        for (int i = 0; i < inputsData.Count / 2; i++)
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
                return 0.2 * Math.Pow(x, 4) + 0.1 * Math.Pow(x, 3) - x * x + 1.0;
            case 6:
                return Math.Tanh(x);
            case 7:
                return Math.Sinh(x);
            default:
                return x;
        }

    }
    private void Generate_Data()
    {
        for (int i = 0; i < data_size; i++)
        {
            double input = Functions.RandomGaussian(0, dataStddev);
            double label = Function(input);
            inputsData.Add(new double[] { input });
            labelsData.Add(new double[] { label });
        }     
    }

    private void Normalize_Inputs_and_Labels()
    {
        if (normalization == NormlizationType.None)
            return;

        for (int i = 0; i < inputsData.Count ; i++)
        {
            inputsNormalizer.OptimizeNormalizer(inputsData[i]);
        }
        for (int i = 0; i < inputsData.Count; i++)
        {
            if(normalization == NormlizationType.ZeroOne)
                inputsNormalizer.Normalize01(inputsData[i], false);
            else if(normalization == NormlizationType.MinusOneOne)
                inputsNormalizer.NormalizeMinusOneOne(inputsData[i], false);
        }
        for (int i = 0; i < labelsData.Count; i++)
        {
            labelsNormalizer.OptimizeNormalizer(labelsData[i]);
        }
        for (int i = 0; i < labelsData.Count; i++)
        {
            if (normalization == NormlizationType.ZeroOne)
                labelsNormalizer.Normalize01(labelsData[i], false);
            else if (normalization == NormlizationType.MinusOneOne)
                labelsNormalizer.NormalizeMinusOneOne(labelsData[i], false);
        }
    }
    private void TrainNetwork()
    {
         double data_acc = 0;

        for (int i = 0; i < inputsData.Count / 2; i++)
        {

            data_acc += net.BackPropagation(inputsData[i], labelsData[i]);
            if(i%batch_size == 0)
                net.OptimiseParameters(learn_rate, momentum, regularization);
        }

        epoch++;

        data_acc /= inputsData.Count/2;
        data_acc = (1 - data_acc) * 100;
        train_accuracy = data_acc.ToString("0.000") + "%";
    }
    private void TestAccuracy()
    {    
        double test_acc = 0;
        testDots.Clear();
        for (int i = inputsData.Count/2; i < inputsData.Count; i++)
        {
            
            double[] inps = inputsData[i];
            double[] outs = net.ForwardPropagation(inps);

            double[] labels = labelsData[i];

            testDots.Add((inps[0], outs[0]));

            double error_on_sample = Functions.Cost.MeanSquare(outs[0], labels[0]);
            
            error_on_sample = (1.0 - error_on_sample) * 100;
            test_acc += error_on_sample;
        }
        test_acc /= inputsData.Count/2;
        test_accuracy = test_acc.ToString("0.000") + "%";
    }

    private void OnDrawGizmos()
    {
        float size = 0.05f;
        Gizmos.color = Color.white;//draw real
        foreach (var dot in targetDots)
        {
            Vector3 pos = new Vector3((float)dot.Item1, (float)dot.Item2, 0);
            Gizmos.DrawCube(pos, Vector3.one * size);
        }



        Gizmos.color = test_color;
        foreach (var dot in testDots)
        {
            Vector3 pos = new Vector3((float)dot.Item1, (float)dot.Item2, 0);
            Gizmos.DrawSphere(pos, size * 0.6f);
        }
        
    }
    
}
public enum NormlizationType
{
    None,
    ZeroOne,
    MinusOneOne,
}