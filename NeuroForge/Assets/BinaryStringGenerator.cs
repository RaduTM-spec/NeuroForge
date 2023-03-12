using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BinaryStringGenerator : MonoBehaviour
{
    static public int[] string01 = new int[4];
    public int[] outs = new int[4];
    private void FixedUpdate()
    {
        for (int i = 0; i < string01.Length; i++)
        {
            string01[i] = Functions.RandomRange(0, 2);
            outs[i] = string01[i];
        }
    }
}
