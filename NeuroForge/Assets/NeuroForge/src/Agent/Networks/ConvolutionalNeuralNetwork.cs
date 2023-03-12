using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class ConvolutionalNeuralNetwork : ScriptableObject
{
    [SerializeField] private NeuralNetwork network;
    [SerializeField] private int convolutionLevel;
    [SerializeField] private KernelType kernelType;
    [SerializeField] private PoolType poolType;

    
    public ConvolutionalNeuralNetwork(
            int inp_width, int inp_height, int outputs,
            int hiddenUnits, int hiddenLayers,
            int convolutionLvl,
            bool createAsset, string name = "cnn",
            KernelType kernel = KernelType.ThreeByThree, 
            PoolType pooling = PoolType.Max,
            ActivationType activType = ActivationType.Relu,
            InitializationType initType = InitializationType.He, 
            LossType lossFunc = LossType.CrossEntropy)
    {
        convolutionLevel = convolutionLvl;
        kernelType = kernel;
        poolType = pooling;

        // Calculate how large will be the flat image, based on all 3 from the top
        int flat_inp_size = 10;

        network = new NeuralNetwork(flat_inp_size, outputs, hiddenUnits, hiddenLayers, activType, ActivationType.SoftMax, lossFunc, initType, false, "none");


        if (createAsset)
        {
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
        }
    }

    public int Forward(double[,] input_image)
    {
        for (int i = 0; i < convolutionLevel; i++)
        {
            Pad(input_image);
            Filter(input_image);
            Pool(input_image);
        }
        double[] flat_input = input_image.Cast<double>().ToArray();
        double[] outputs = network.Forward(flat_input);
        return Functions.ArgMax(outputs);
    }
    public double Backward(double[,] image, int label)
    {
        return 0;
    }
    public void OptimStep(float learnRate, float momentum, float regularization) => 
        network.OptimStep(learnRate, momentum, regularization);

    private void Pad(double[,] image)
    {
        double[,] padded_image = kernelType == KernelType.ThreeByThree ?
            new double[image.GetLength(0) + 2, image.GetLength(1) + 2] :
            new double[image.GetLength(0) + 4, image.GetLength(1) + 4];

        if(kernelType == KernelType.ThreeByThree)
        {

        }
        else // FiveByFive
        {

        }

        image = padded_image;
    }
    private void Filter(double[,] image)
    {
     
    } 
    private void Pool(double[,] image)
    {
        double[,] pooled_image = new double[image.GetLength(0), image.GetLength(1)];

        for (int i = 0; i < image.GetLength(0); i+=2)
        {
            for (int j = 0; j < image.GetLength(1); j+=2)
            {
                double[] pool = new double[4];
                pool[0] = image[i, j];
                pool[1] = image[i, j + 1];
                pool[2] = image[i + 1, j];
                pool[3] = image[i + 1, j + 1];
                pooled_image[i / 2, j / 2] = poolType == PoolType.Max ?
                                            pool.Max() :
                                            pool.Average();
            }
        }
        image = pooled_image;
    }


    private static double[,] kernel3x3 = new double[,]
    {
        {0,1,0},
        {1,2,1},
        {0,1,0}
    };
    private static double[,] kernel5x5 = new double[,]
    {
        {0,1,1,1,0},
        {1,2,2,2,1},
        {1,2,4,2,1},
        {1,2,2,2,1},
        {0,1,1,1,0},
    };
}
