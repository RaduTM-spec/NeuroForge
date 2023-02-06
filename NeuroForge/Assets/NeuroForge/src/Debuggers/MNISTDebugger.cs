using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
using System.IO;
using UnityEditor.Build.Content;
using System.Linq;
using UnityEditor.PackageManager;

public class MNISTDebugger : MonoBehaviour
{
    public NeuralNetwork net;
    public int hiddenUnits = 64;
    public int layerNum = 2;
    [Range(1e-5f,1e-2f)] public float learnRate = 0.001f;
    public float momentum = 0.9f;
    public float regularization = 1e-5f;
    public InitializationType initType = InitializationType.He;
    public ActivationType activation = ActivationType.Relu;
    public LossType loss = LossType.MeanSquare;
    public int TrainSamplesPerDigit = 64;
    public int TestSamplesPerDigit = 64;

    public int Epoch = 0;
    public string TrainAccuracy;
    public string TestAccuracy;
    public int Correct = 0;
    public int Wrong = 0;
    public bool train = true;
    public bool test = true;

    public Dictionary<int, List<float[]>> batch;
    public Dictionary<int, List<float[]>> testData;
    void Awake()
    {
        if(net == null)
            net = new NeuralNetwork(784, 10, hiddenUnits, layerNum, activation, ActivationType.SoftMax, loss, initType,
            true, "MNISTNetwork");

        
    }
    private void Update()
    {
        if(train)Train();
        if(test)Test();
    }
    void LoadRandomBatch()
    {
        batch = new Dictionary<int, List<float[]>>();
        
        string trainPath = "C:\\Users\\X\\Desktop\\TRAIN\\";
        for (int i = 0; i < 10; i++)
        {
            batch.Add(i, new List<float[]>());
            trainPath += i;
            string[] imagesPaths = Directory.GetFiles(trainPath,"*.jpg", SearchOption.TopDirectoryOnly);
            
            for (int j = 0; j < TrainSamplesPerDigit; j++)
            {
                int randomPos = (int)(UnityEngine.Random.value * imagesPaths.Length);
                float[] imgPix = LoadTexture(imagesPaths[randomPos]).GetPixels().Select(x => x.grayscale).ToArray();
                batch[i].Add(imgPix);
            }
            trainPath = trainPath.Substring(0, trainPath.Length - 1);
        }
    }
    void Train()
    {
        Epoch++;
        LoadRandomBatch();

        double avgError = 0;
        foreach (var digit in batch)
        {
            double[] labels = Enumerable.Repeat(0.0, 10).ToArray();
            labels[digit.Key] = 1.0;

            foreach (var inputs in digit.Value)
            {
                double[] inpts = inputs.Select(x => (double)x).ToArray();
                avgError += net.BackPropagation(inpts, labels);
            }
            net.OptimizeParameters(learnRate, momentum, regularization);
        }
        avgError /= (10 * TrainSamplesPerDigit);

        TrainAccuracy = ((1.0 -  avgError) * 100).ToString("00.000") + "%";
    }


    void LoadRandomTest()
    {
        if (!test)
            return;
        testData = new Dictionary<int, List<float[]>>();

        string testPath = "C:\\Users\\X\\Desktop\\TEST\\";
        for (int i = 0; i < 10; i++)
        {
            testData.Add(i, new List<float[]>());
            testPath += i;
            string[] imagesPaths = Directory.GetFiles(testPath, "*.jpg", SearchOption.TopDirectoryOnly);

            for (int j = 0; j < TestSamplesPerDigit; j++)
            {
                int randomPos = (int)(UnityEngine.Random.value * imagesPaths.Length);
                float[] imgPix = LoadTexture(imagesPaths[randomPos]).GetPixels().Select(x => x.grayscale).ToArray();
                testData[i].Add(imgPix);
            }
            testPath = testPath.Substring(0, testPath.Length - 1);
        }
    }
    void Test()
    {
        LoadRandomTest();

        Correct = 0;
        Wrong = 0;
        foreach (var digit in testData)
        {
            double[] labels = Enumerable.Repeat(0.0, 10).ToArray();
            labels[digit.Key] = 1.0;

            foreach (var inputs in digit.Value)
            {
                double[] inpts = inputs.Select(x => (double)x).ToArray();
                double[] outs = net.ForwardPropagation(inpts);

                
                Functions.Activation.OneHot(outs);
                bool isCorrect = true;
                for (int i = 0; i < outs.Length; i++)
                {
                    if (outs[i] != labels[i])
                        isCorrect = false;
                }
                if (isCorrect)
                    Correct++;
                else
                    Wrong++;
            }
        }
        TestAccuracy = ((float)Correct / (float)(Correct + Wrong) * 100f).ToString("00.000") + "%";
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
