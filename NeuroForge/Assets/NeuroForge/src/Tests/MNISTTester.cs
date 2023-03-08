using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using static UnityEngine.Mesh;
using System.Linq;
using NeuroForge;

public class MNISTTester : MonoBehaviour
{
    public NeuralNetwork model;
    public int TestSamplesPerDigit = 64;
    public Dictionary<int, List<float[]>> testData;

    [Space] public int digitLabel = -1;
    public int netPrediction = -1;
    void Start()
    {
        GenerateTestData();
        StartCoroutine(Test());
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

            for (int j = 0; j < TestSamplesPerDigit; j++)
            {
                float[] imgPix = LoadTexture(imagesPaths[j]).GetPixels().Select(x => x.grayscale).ToArray();
                testData[i].Add(imgPix);
            }
            testPath = testPath.Substring(0, testPath.Length - 1);
        }
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
    IEnumerator Test()
    {
        yield return new WaitForSeconds(.5f);


        // Select random image
        digitLabel = (int)(UnityEngine.Random.value * 10f);
        
        List<float[]> allDigitImgs = testData[digitLabel];
        int randImg = (int)UnityEngine.Random.value * allDigitImgs.Count;
        double[] input = allDigitImgs[randImg].Select(x => (double)x).ToArray();

        /*// Render image
        Texture2D texture = new Texture2D(28, 28);
        Color[] colors = input.Select(x => new Color((float)x, (float)x, (float)x, 1)).ToArray();
        texture.SetPixels(colors);
        Material mat = new Material(Shader.Find("Unlit/Texture"));
        mat.mainTexture = texture;
        GameObject obj = new GameObject();
        MeshRenderer renderer = obj.AddComponent<MeshRenderer>();
        renderer.material = mat;
        MeshFilter meshFilter = obj.AddComponent<MeshFilter>();
        meshFilter.mesh = new Mesh();*/

        // Generate predict
        double[] output = model.Forward(input);
        netPrediction = Functions.ArgMax(output);

        if (netPrediction == digitLabel)
            Debug.Log("Correct");
        else
            Debug.Log("Wrong");

        StartCoroutine(Test());
    }
}
