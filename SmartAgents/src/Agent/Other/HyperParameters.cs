using SmartAgents;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace SmartAgents
{
    public class HyperParameters : MonoBehaviour
    {
        public HiddenLayers networkHiddenLayers = HiddenLayers.OneLarge;
        public ActivationType activationType = ActivationType.Tanh;
        public LossType lossType = LossType.MeanSquare;

        [Space]
        [Range(0.000001f, 1f)] public float learnRate = 0.001f;
        [Range(0.000001f, 1f)] public float momentum = 0.9f;
        [Range(0.000001f, 0.1f)] public float regularization = 0.001f;
        [Range(0.000001f, 1f)] public float discountFactor = 0.99f;
        [Range(0.000001f, 1f)] public float gaeFactor = 0.95f;
        [Range(0.000001f, 1f)] public float clipFactor = 0.2f;

        [Space]
        [SerializeField] private MemorySize memorySize = MemorySize.size1024;    
        [SerializeField] private MiniBatchSize miniBatchSize = MiniBatchSize.size64;
        [Min(1)] public int epochs = 10;
        [Min(0)] public int maxStep = 0;

        [HideInInspector] public int memory_size = 1024;
        [HideInInspector] public int mini_batch_size = 64;
        private void Awake()
        {
            memory_size = (int)Math.Pow(2, (int)memorySize + 8);
            mini_batch_size = (int)Math.Pow(2, (int)miniBatchSize + 5);
            if (mini_batch_size > memory_size)
            {
                mini_batch_size = memory_size;
                miniBatchSize = (MiniBatchSize)(int)memorySize + 3;
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