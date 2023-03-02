using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class XORInputGenerator : MonoBehaviour
{
    static public int[] inputs = new int[2];

    private void Update()
    {
        inputs[0] = Functions.RandomValue() < .5f ? 1 : 0;
        inputs[1] = Functions.RandomValue() < .5f ? 1 : 0;
    }
}
