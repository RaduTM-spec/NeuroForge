using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace SmartAgents
{
    public class HyperParameters : MonoBehaviour
    {
        [Header("Network")]
        public HiddenLayers networkHiddenLayers = HiddenLayers.OneLarge;
        [Range(0.000001f, 1f)] public float learnRate = 0.1f;
        [Range(0f, 1f)] public float momentum = 0.9f;
        [Range(0f, 0.1f)] public float regularization = 0.001f;
        public ActivationType activationType = ActivationType.Tanh;
        public LossType lossType = LossType.MeanSquare;

        [Header("Training")]
        [Range(0, 100_000)] public int memoryCapacity = 0;
        [Range(0, 1f)] public float discountFactor = 0.1f;
        [Min(0)] public int maxStep = 0;

        [Space]
        public AnimationCurve progressChart = new AnimationCurve();
        public int epoch = 0;
        public string accuracy;
        

        public void ClearProgressChart()
        {
            for (int i = 0; i < progressChart.length; i++)
            {
                progressChart.RemoveKey(i);
            }
        }
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
}