using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NEATTestResultGenome : MonoBehaviour
{
    public Genome genome;
    public int iterations = 1000;
    public float error;

    // Update is called once per frame


    private void Start()
    {
        for (int i = 0; i < iterations; i++)
        {
            double[] input = GetInputs();
            double label = (int)input[0] ^ (int)input[1];
            
            ///>>> TO COMPLETE
            int output = genome.GetDiscreteActions(input)[0];
            error += Mathf.Abs((float)label - output);
        }
    }
    private double[] GetInputs()
    {
        double[] inputs = new double[2];
        inputs[0] = FunctionsF.RandomValue() < .5f ? 0 : 1;
        inputs[1] = FunctionsF.RandomValue() < .5f ? 0 : 1;
        return inputs;
    }

}
