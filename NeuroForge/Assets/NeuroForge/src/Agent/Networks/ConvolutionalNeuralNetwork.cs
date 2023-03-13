using NeuroForge;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class ConvolutionalNeuralNetwork : ScriptableObject
    {
        // This CNN implementation was not generalized because of large number of parameter settings
        // Thus, was standardized to a static kernel3x3, max pooling, ReLU activation, CrossEntropy Loss, HE initialization
        [SerializeField] private NeuralNetwork network;
        [SerializeField] private int convolutionLevel;

        int[,] kernel = new int[3, 3]
        {
        { -1, -1, -1 },
        { -1,  8, -1 },
        { -1, -1, -1 }
        };

        public ConvolutionalNeuralNetwork(int inp_width, int inp_height, int outputs, int hidUnits, int hidLayers, int convolutionLvl, bool createAsset, string name = "cnn")
        {
            convolutionLevel = convolutionLvl;

            for (int i = 0; i < convolutionLvl; i++)
            {
                inp_width /= 2;
                inp_height /= 2;
            }

            network = new NeuralNetwork(inp_width * inp_height, outputs, hidUnits, hidLayers,
                                        ActivationType.Relu, ActivationType.SoftMax, LossType.CrossEntropy,
                                        InitializationType.He, true, "cnn_aux");


            if (createAsset)
            {
                Debug.Log(name + " was created!");
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
                AssetDatabase.SaveAssets();
            }
        }

        public int Forward(float[,] input_image)
        {
            for (int i = 0; i < convolutionLevel; i++)
            {
                Pad(ref input_image);
                Filter(ref input_image);
                RescaleFilteredImage(ref input_image);
                Pool(ref input_image);
            }
            double[] flat_input = input_image.Cast<double>().ToArray();
            double[] outputs = network.Forward(flat_input);
            return Functions.ArgMax(outputs);
        }
        public double Backward(float[,] input_image, int label)
        {
            for (int i = 0; i < convolutionLevel; i++)
            {
                Pad(ref input_image);
                Filter(ref input_image);
                RescaleFilteredImage(ref input_image);
                Pool(ref input_image);
            }

            double[] flat_inputs = input_image.Cast<double>().ToArray();
            double[] labels = new double[network.GetNoOutputs()];
            labels[label] = 1;

            double error = network.Backward(flat_inputs, labels);
            return error;
        }
        public void GradClipNorm(float threshold) => network.GradClipNorm(threshold);
        public void OptimStep(float learnRate, float momentum, float regularization) => network.OptimStep(learnRate, momentum, regularization);


        // Convolution Methods
        private void Pad(ref float[,] image)
        {
            // works 100%
            float[,] padded_image = new float[image.GetLength(0) + 2, image.GetLength(1) + 2];

            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    padded_image[i + 1, j + 1] = image[i, j];
                }
            }

            image = padded_image;
        }
        private void Filter(ref float[,] image)
        {
            // drop is 1
            // Filtering does not affect the dimension of the final image (only pooling)
            // Image is padded. When applying kernel, f_img will be 2 less for each dimension
            float[,] filtered_image = new float[image.GetLength(0) - 2, image.GetLength(1) - 2];

            // Parse each pixel
            for (int i = 1; i < image.GetLength(0) - 1; i += 1)
            {
                for (int j = 1; j < image.GetLength(1) - 1; j += 1)
                {
                    // Filter-up
                    float sum = 0;
                    for (int k_i = 0; k_i < kernel.GetLength(0); k_i++)
                    {
                        for (int k_j = 0; k_j < kernel.GetLength(1); k_j++)
                        {
                            sum += image[i - 1 + k_i, j - 1 + k_j] * kernel[k_i, k_j];
                        }
                    }

                    filtered_image[i - 1, j - 1] = sum;
                }
            }

            image = filtered_image;
        }
        private void RescaleFilteredImage(ref float[,] image)
        {
            float max_val = float.MinValue;
            float min_val = float.MaxValue;

            // Find min & max
            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    if (image[i, j] > max_val)
                        max_val = image[i, j];
                    if (image[i, j] < min_val)
                        min_val = image[i, j];
                }
            }

            // Scale [0,1]
            float delta = max_val - min_val;
            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    image[i, j] = (image[i, j] - min_val) / delta;
                }
            }
        }
        private void Pool(ref float[,] image)
        {
            float[,] pooled_image = new float[image.GetLength(0) / 2, image.GetLength(1) / 2];
            for (int i = 0; i < pooled_image.GetLength(0); i++)
            {
                for (int j = 0; j < pooled_image.GetLength(1); j++)
                {
                    float[] local_pool = new float[4];
                    local_pool[0] = image[i * 2, j * 2];
                    local_pool[1] = image[i * 2, j * 2 + 1];
                    local_pool[2] = image[i * 2 + 1, j * 2];
                    local_pool[3] = image[i * 2 + 1, j * 2 + 1];

                    pooled_image[i, j] = local_pool.Max();
                }
            }

            image = pooled_image;
        }



    }

}
