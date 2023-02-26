using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using static UnityEngine.Mesh;

public class MNISTTrain : MonoBehaviour
{
    public PPOActor network;
    public int miniBatchSize = 64;
    Dictionary<int, List<float[]>> trainData;
    Dictionary<int, List<float[]>> testData;

    [Space]
    public float learnRate = 0.0003f;
    public float momentum = 0.9f;
    public float regularization = 0.00005f;

    public string trainAcc;
    public string testAcc;

    private void Awake()
    {
        if(network == null)
        {
            network = new PPOActor(784, new int[] { 10 }, 50, 1, ActivationType.Relu, InitializationType.He);
        }
    }

    public void Update()
    {
        GenerateTrainData();
        GenerateTestData();

        Train();
        Test();
    }
    void GenerateTrainData()
    {
        trainData = new Dictionary<int, List<float[]>>();

        string trainPath = "C:\\Users\\X\\Desktop\\TRAIN\\";
        for (int i = 0; i < 10; i++)
        {
            trainData.Add(i, new List<float[]>());
            trainPath += i;
            string[] imagesPaths = Directory.GetFiles(trainPath, "*.jpg", SearchOption.TopDirectoryOnly);

            for (int j = 0; j < miniBatchSize; j++)
            {

                float[] imgPix = LoadTexture(Functions.RandomIn(imagesPaths)).GetPixels().Select(x => x.grayscale).ToArray();
                trainData[i].Add(imgPix);
            }
            trainPath = trainPath.Substring(0, trainPath.Length - 1);
        }
    }
    void GenerateTestData()
    {
        testData = new Dictionary<int, List<float[]>>();

        string testPath = "C:\\Users\\X\\Desktop\\TEST\\";
        for (int i = 0; i < 10; i++)
        {
            testData.Add(i, new List<float[]>());
            testPath += i;
            string[] imagesPaths = Directory.GetFiles(testPath, "*.jpg", SearchOption.TopDirectoryOnly);

            for (int j = 0; j < miniBatchSize; j++)
            {
                float[] imgPix = LoadTexture(imagesPaths[j]).GetPixels().Select(x => x.grayscale).ToArray();
                testData[i].Add(imgPix);
            }
            testPath = testPath.Substring(0, testPath.Length - 1);
        }
    }
    void Train()
    {
        double err = 0.0;
        int count = 0;
        foreach (var digit in trainData)
        {
            double[] labels = new double[10];
            labels[digit.Key] = 1;
            foreach (var sample in digit.Value)
            {
                // MSE
                double[] input = sample.Select(x => (double)x).ToArray();
                double[] prediction = network.DiscreteForwardPropagation(input).Item1;
               
                double[] losses = new double[10];
               
                for (int i = 0; i < 10; i++)
                {
                    losses[i] = Functions.Cost.MeanSquareDerivative(prediction[i], labels[i]);
                    err += .5 * Functions.Cost.MeanSquare(prediction[i], labels[i]);
                }
               
                count++;
                network.BackPropagation(input, losses);
                network.OptimiseParameters(learnRate, momentum, regularization);
            }         
        }
        trainAcc = ((1.0 - err / count) * 100).ToString("0.000");
    }
    void Test()
    {
        double err = 0.0;
        int count = 0;
        foreach (var digit in testData)
        {
            double[] labels = new double[10];
            labels[digit.Key] = 1;
            foreach (var sample in digit.Value)
            {
                double[] input = sample.Select(x => (double)x).ToArray();
                double[] prediction = network.DiscreteForwardPropagation(input).Item1;

                for (int i = 0; i < 10; i++)
                {
                    err += 0.5 * Functions.Cost.MeanSquare(prediction[i], labels[i]);
                }

                count++;
            }
        }
        testAcc = ((1.0 - err / count) * 100).ToString("0.000") + "%";
    }
    private Texture2D LoadTexture(string filePath)
    {
        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(28, 28);
            tex.LoadImage(fileData);
        }
        return tex;
    }
}
