using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NEATTestResultGenome : MonoBehaviour
{
    public Genome genome;
    public int iterations = 1000;
    public string accuracy;
    // Update is called once per frame


    private void Start()
    {
        float error = 0;
        for (int i = 0; i < iterations; i++)
        {
            double[] input = GetInputs();
            double XOR = (int)input[0] ^ (int)input[1];
            
            ///>>> TO COMPLETE
            float output = genome.GetContinuousActions(input)[0];
            error += Mathf.Abs((float)XOR - output);
        }
        error /= iterations;
        accuracy = ((1f - error) * 100f).ToString("0.000") + "%";
    }
    private double[] GetInputs()
    {
        double[] inputs = new double[2];
        inputs[0] = FunctionsF.RandomValue() < .5f ? 0 : 1;
        inputs[1] = FunctionsF.RandomValue() < .5f ? 0 : 1;
        return inputs;
    }

}
