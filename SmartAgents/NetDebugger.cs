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

public class NetDebugger : MonoBehaviour
{
    public ArtificialNeuralNetwork Network;
    public Memory memory;

    [Space] public bool findRewardinMem = false;
    [Space]
    public bool loadMemory = true;
    public bool trainNet = false;
    public bool showOuts = false;


    [SerializeField] AnimationCurve chart = new AnimationCurve();


    List<double[]> inputsData = new List<double[]>();
    List<double[]> labelsData = new List<double[]>();
    List<float> progress = new List<float>();
    
    private void Awake()
    {
        InitNetwork();
        
    }
    private void Start()
    {
        if(findRewardinMem) 
             FindRewardInMemory();
        if(loadMemory)
            FillInputs_FillLabels();
        if (trainNet)
            TrainNetwork();
        if (showOuts)
            ShowOuts();
    }

    private void FillInputs_FillLabels()
    {
        foreach (var item in memory.records)
        {
            inputsData.Add(item.state.Concat(item.action).ToArray());
            labelsData.Add(new double[] {item.reward});
        }

       /* return;
        for (int i = 1; i <= 100; i++)
        {
            double pointY = Random.value * 4;
            inputsData.Add(new double[] { pointY, i });
            if(pointY > Mathf.Log(i))
            {
                labelsData.Add(new double[] { 1 });
            }
            else
            {
                labelsData.Add(new double[] { 0 });
            }
            
        }*/
    }

    private void InitNetwork()
    {
        if (Network != null)
            return;

        Network = new ArtificialNeuralNetwork(2, 1, HiddenLayers.OneLarge, ActivationType.Tanh, ActivationType.Tanh, LossType.MeanSquare, GetName());

        string GetName()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<ArtificialNeuralNetwork>("Assets/TestCriticNN#" + id + ".asset") != null)
                id++;
            return "TestCriticNN#" + id;
        }
    }
    private void TrainNetwork()
    {
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            double epochErr = 0;
            for (int i = 0; i < inputsData.Count; i++)
            {
                epochErr += Network.BackPropagation(inputsData[i], labelsData[i]);
            }
            Debug.Log("Epoch [" + epoch + "] Error -> " + epochErr/inputsData.Count);
            progress.Add((float)epochErr/inputsData.Count);

            Network.UpdateParameters(0.1f, 0.9f, 0.001f);
        }
    }
    private void ShowOuts()
    {
        //Clear Chart
        for (int i = 0; i < chart.length; i++)
        {
            chart.RemoveKey(i);
        }

        double totalAccuracy = 0;
        for (int i = 0; i < inputsData.Count; i++)
        {
            double[] inps = inputsData[i];
            double[] outs = Network.ForwardPropagation(inps);
            double[] labels = labelsData[i];
            StringBuilder sb = new StringBuilder();


            double accuracy = 0;
            for (int j = 0; j < outs.Length; j++)
            {
                accuracy += Mathf.Abs((float)(outs[j] - labels[j]));
            }
            accuracy /= outs.Length;
            accuracy = (1 - accuracy) * 100;
            totalAccuracy+= accuracy;

            sb.Append("Accuracy: " + accuracy.ToString("0.000") + "% | ");
            sb.Append("Count: " + i + " | ");
            sb.Append("inputs:[ ");
            foreach (var item in inps)
            {
                sb.Append(item.ToString("0.000") + ", ");
            }
            sb.Remove(sb.Length - 2, 1);
            sb.Append("] -------------> outputs:[ ");
            foreach (var item in outs)
            {
                sb.Append(item.ToString("0.000") + ", ");
            }
            sb.Remove(sb.Length - 2, 1);

            sb.Append("] <===> labels:[ ");
            foreach (var item in labels)
            {
                sb.Append(item.ToString("0.000") + ", ");
            }
            sb.Remove(sb.Length - 2, 1);
            sb.Append("]");


            Debug.Log(sb.ToString());
        }
        Debug.Log("ACCURACY ON DATASET: " + totalAccuracy / inputsData.Count + "%");
    }


    private void FindRewardInMemory()
    {
        for (int i = 0; i < memory.records.Count; i++)
        {
            if (memory.records[i].reward != 0)
                Debug.Log("Found reward on Element: "  + i + " / " + memory.records.Count + "  ---> Reward# " + memory.records[i].reward);
        }
    }

   
}
