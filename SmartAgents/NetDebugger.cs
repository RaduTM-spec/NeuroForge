using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SmartAgents;
public class NetDebugger : MonoBehaviour
{
    public ArtificialNeuralNetwork actor;
    public ArtificialNeuralNetwork critic;
    public Memory memory;

    List<(float, float, float, float, float, float)> obsLabels;
    List<(float, float)> actLabels;
    private void Start()
    {
        
    }
}
