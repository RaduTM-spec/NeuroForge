using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class XORInputGenerator : MonoBehaviour
{
    static public int[] inputs = new int[2];

    int step = 0;
    private void FixedUpdate()
    {
       switch(step % 4)
        {
            case 0:
                inputs[0] = 0;
                inputs[1] = 0;
                break;
            case 1:
                inputs[0] = 1;
                inputs[1] = 0;
                break;
            case 2:
                inputs[0] = 0;
                inputs[1] = 1;
                break;
            case 3:
                inputs[0] = 1;
                inputs[1] = 1;
                break;
        }
        step++;
    }
}
