using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class HyperParameters : MonoBehaviour
{
    public HiddenLayers networkHiddenLayers = HiddenLayers.OneLarge;
    [Space]
    [Range(0, 1f)] public float discountFactor = 0.1f;
    [Range(0f, 1f)] public float learnRate = 0.1f;
    [Range(0f, 1f)] public float momentum = 0.9f;
    [Range(0f, 1f)] public float regularization = 0.01f;
    public ActivationType activationType = ActivationType.Tanh;//output activation is SoftMax for discrete actions and Tanh for continuous actions
    public LossType lossType = LossType.MeanSquare;//Cross Entropy for discrete actions

}
[CustomEditor(typeof(HyperParameters), true), CanEditMultipleObjects]
class ScriptlessHP : Editor
{
    private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        DrawPropertiesExcluding(serializedObject, _dontIncludeMe);

        serializedObject.ApplyModifiedProperties();
    }
}
