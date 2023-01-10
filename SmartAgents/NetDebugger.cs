using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SmartAgents;
using UnityEditor;
using Unity.VisualScripting;
using System.Net.Sockets;
using System.Linq;
using UnityEngine.UIElements;
using System.Text;
using System;
using Newtonsoft.Json.Linq;

public class NetDebugger : MonoBehaviour
{
    public ArtificialNeuralNetwork net;
    public int hiddenUnits = 64;
    public int layerNum = 2;
    public ActivationType activation = ActivationType.Relu;

    [Space]
    [Range(0.1f,10f)]public double stddev = 1;
    public int batch_size = 100;
    public int epoch = 0;

    [Space]
    public string train_accuracy;
    public string test_accuracy; public Color test_color = Color.red;


    [Space]
    [Range(0.00001f,0.1f)]public float learn_rate = 0.1f;
    [Range(0.00000f, 0.1f)] public float regularization = 0.001f;
    [Range(0.00000f, 1f)] public float momentum = 0.9f;

    List<double[]> inputsData = new List<double[]>();
    List<double[]> labelsData = new List<double[]>();

    List<double[]> inputsTest = new List<double[]>();
    List<double[]> labelsTest = new List<double[]>();

    List<(double, double)> targetDots = new List<(double, double)>();
    List<(double, double)> testDots = new List<(double, double)>();

    private void Start()
    {
        net = new ArtificialNeuralNetwork(1, new int[] {1}, hiddenUnits, layerNum, activation, ActivationType.Tanh, LossType.MeanSquare, true, "test");
        Generate_Data();
        Normalize_Data(inputsData);
        Normalize_Data(labelsData);
        Normalize_Data(inputsTest);
        Normalize_Data(labelsTest);
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
    double Function(double x)
    {
        //return   x / (1 + Mathf.Exp((float)-x));//silu
        //return Math.Cos(x);
        return Math.Sin(x);
        //return x / (1 + x * x);
        //return x * x * x / Math.Pow(2, x);
        //return 0.2 * Math.Pow(x, 4) + 0.1 * Math.Pow(x, 3) - x * x + 1.0;
        //return 3 / x * Math.Exp(-0.5 * x);
    }
    private void Generate_Data()
    {
        for (int i = 0; i < batch_size; i++)
        {
            double input = Functions.RandomGaussian(0,stddev);
            double label = Function(input);
            inputsData.Add(new double[] { input });
            labelsData.Add(new double[] { label });
        }
        for (int i = 0; i < batch_size; i++)
        {
            double input = Functions.RandomGaussian(0,stddev);
            double label = Function(input);
            inputsTest.Add(new double[] { input });
            labelsTest.Add(new double[] { label });
        }
             
        
    }
    private void Normalize_Data(List<double[]> data)
    {
        double[] mins = new double[inputsData[0].Length];
        double[] maxs = new double[inputsData[0].Length];
        for (int i = 0; i < mins.Length; i++)
        {
            mins[i] = double.MaxValue;
            maxs[i] = double.MinValue;
        }

        //Find min and max
        for (int i = 0; i < data.Count; i++)
        {
            for (int j = 0; j < data[i].Length; j++)
            {
                if (data[i][j] < mins[j])
                    mins[j] = data[i][j];
                else if (data[i][j] > maxs[j])
                    maxs[j] = data[i][j];
            }
        }

        //Normalize to (-1,1)
        for (int i = 0; i < data.Count; i++)
        {
            for (int j = 0; j < data[i].Length; j++) // Number of outputs
            {
                data[i][j] =  2 * (data[i][j] - mins[j]) / (maxs[j] - mins[j]) - 1;
            }
        }

    }
    private void TrainNetwork()
    {
        double data_acc = 0;

        /*System.Threading.Tasks.Parallel.For(0, inputsData.Count, i =>
        {
            double[] outs = net.ForwardPropagation(inputsData[i]);
            data_acc += net.BackPropagation(inputsData[i], labelsData[i]);             
        });
        net.OptimizeParameters(learn_rate, momentum, regularization, true);
*/
         for (int i = 0; i < inputsData.Count; i++)
           {
              double[] outs = net.ForwardPropagation(inputsData[i]);
              data_acc += net.BackPropagation(inputsData[i], labelsData[i]);
              if (i%10 == 0) //minibatch
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
        for (int i = 0; i < inputsTest.Count; i++)
        {
            double[] inps = inputsTest[i];
            double[] outs = net.ForwardPropagation(inps);
            double[] labels = labelsTest[i];

            testDots.Add((inps[0], outs[0]));

            double accuracy_on_sample = Mathf.Abs((float)(outs[0] - labels[0]));
            
            accuracy_on_sample = (1 - accuracy_on_sample) * 100;
            test_acc += accuracy_on_sample;
        }
        test_acc /= inputsTest.Count;
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
