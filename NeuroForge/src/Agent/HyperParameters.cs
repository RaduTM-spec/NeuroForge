using NeuroForge;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    public class HyperParameters : MonoBehaviour
    {
        [Range(16,128)]public int hiddenUnits = 64;
        [Range(1,5)]public int layersNumber = 2;
        public InitializationType initializationType = InitializationType.He;
        public ActivationType activationType = ActivationType.Relu;

        [Header("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")]
        [Range(0.00001f, 0.01f), Tooltip("alpha")] public float actorLearnRate = 0.0003f;
        [Range(0.00001f, 0.1f), Tooltip("alpha'")] public float criticLearnRate = 0.001f;
        [Range(0.0f, 1.0f), Tooltip("mu")] public float momentum = 0.9f;
        [Range(0.0f, 0.1f), Tooltip("beta")] public float regularization = 0.0001f;

        [Header("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")]
        [Range(0.8f, 0.995f), Tooltip("gamma")] public float discountFactor = 0.99f;
        [Range(0.9f, 0.95f), Tooltip("lambda")] public float gaeFactor = 0.95f;
        [Range(0.1f, 0.3f), Tooltip("epsilon")] public float clipFactor = 0.2f;
        [Range(0.0001f, 0.01f), Tooltip("beta")] public float entropyRegularization = 0.0001f;

        [Header("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")]
        [SerializeField] private BatchSize bufferSize = BatchSize.size2048;    
        [SerializeField] private MiniBatchSize batchSize = MiniBatchSize.size512;
        [Min(1)] public int epochs = 10;

        [HideInInspector] public int buffer_size;
        [HideInInspector] public int batch_size;
        private void Awake()
        {
            buffer_size = (int)Math.Pow(2, (int)bufferSize + 10);
            batch_size = (int)Math.Pow(2, (int)batchSize + 8);
            if (batch_size > buffer_size)
            {
                batch_size = buffer_size;
                batchSize = (MiniBatchSize)(int)bufferSize + 2;
            }
        }
    }
    public enum BatchSize
    {
        size1024,
        size2048,
        size4096,
        size8192,
        size16384,
        size32768,
        size65536,
        size131072,
        size262144,
    }
    public enum MiniBatchSize
    {
        size256,
        size512,
        size1024,
        size2048,
        size4096,
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