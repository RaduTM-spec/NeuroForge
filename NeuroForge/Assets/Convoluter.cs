using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Windows;

public class Convoluter : MonoBehaviour
{
    [SerializeField] public int convolutionLevel = 1;
    public int drop = 1; // recommended on 1

    [SerializeField] private KernelType kernelType = KernelType.Sharp_3x3;
    [SerializeField] private PoolType poolType = PoolType.Max;
    public Image img;
    public Image conv;
    int[,] kernel = Functions.Image.Kernel.kernel3x3_sharp;

    // Start is called before the first frame update
    void Start()
    {
        // Convert img sprite to [,]
        float[,] input_image = new float[img.sprite.texture.width, img.sprite.texture.height];
        float[] flat_image = img.sprite.texture.GetPixels().Select(x => x.grayscale).ToArray();
        int ind = 0;
        for (int i = 0; i < input_image.GetLength(0); i++)
        {
            for (int j = 0; j < input_image.GetLength(1); j++)
            {
                input_image[i, j] = flat_image[ind++];
            }
        }

        for(int i = 0 ; i < convolutionLevel ; i++)
        {
            Pad(ref input_image);
            Filter(ref input_image);
            RescaleFilteredImage(ref input_image);
            Pool(ref input_image);
        }
       

        

        Color[] pixelsC = input_image.Cast<float>().Select(x => new Color(x,x,x)).ToArray();

        Debug.Log("Original image size: " + flat_image.Length);
        Debug.Log("After convolution size: " + pixelsC.Length);

        var x = new Texture2D(input_image.GetLength(0), input_image.GetLength(1));
        x.SetPixels(pixelsC);
        x.Apply();

        conv.sprite = Sprite.Create(x, new Rect(0, 0, input_image.GetLength(0), input_image.GetLength(1)), new Vector2(0.5f, 0.5f));
        conv.sprite.texture.filterMode = FilterMode.Point;

    }

    // THIS REMAINS A GENERALIZED TYPE OF CONVOLUTER

    private void Pad(ref float[,] image)
    {
        // works 100%
        float[,] padded_image = kernelType == KernelType.Sharp_3x3 ?
            new float[image.GetLength(0) + 2, image.GetLength(1) + 2] :
            new float[image.GetLength(0) + 4, image.GetLength(1) + 4];

        int right_down_inc = kernelType == KernelType.Sharp_3x3 ? 1 : 2;

        for (int i = 0; i < image.GetLength(0); i++)
        {
            for (int j = 0; j < image.GetLength(1); j++)
            {
                padded_image[i + right_down_inc, j + right_down_inc] = image[i, j];
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
        for (int i = 1; i < image.GetLength(0) - 1; i += drop)
        {
            for (int j = 1; j < image.GetLength(1) - 1; j += drop)
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
                pooled_image[i, j] = poolType == PoolType.Max ?
                                            local_pool.Max() :
                                            local_pool.Average();
            }
        }

        image = pooled_image;
    }



    
}
