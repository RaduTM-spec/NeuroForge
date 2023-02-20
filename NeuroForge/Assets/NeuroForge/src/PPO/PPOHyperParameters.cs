using NeuroForge;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [DisallowMultipleComponent, AddComponentMenu("NeuroForge/HyperParameters")]
    public class PPOHyperParameters : MonoBehaviour
    {
        [Range(16,128)]public int hiddenUnits = 32;
        [Range(1,5)]public int layersNumber = 2;
        public InitializationType initializationType = InitializationType.He;
        public ActivationType activationType = ActivationType.LeakyRelu;

        [Header("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")]
        [Range(1e-5f, 1e-3f), Tooltip("alpha")] public float actorLearnRate = 0.0003f;                                                                                                         
        [Range(1e-5f, 1e-2f), Tooltip("alpha'")] public float criticLearnRate = 0.001f;                                                                                                        
        [Range(0f, 1f), Tooltip("mu")] public float momentum = 0.9f;                                                                                                                           
        [Range(0f, 1e-1f), Tooltip("beta")] public float regularization = 0.00001f;                                                                                                            
                                                                                                                                                                                               
        [Header("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")]
        [Range(0.8f, 0.995f), Tooltip("gamma")] public float discountFactor = 0.9f;                                                                                                             
        [Range(0.9f, 0.95f), Tooltip("lambda")] public float gaeFactor = 0.95f;                                                                                                                 
        [Range(0.1f, 0.3f), Tooltip("epsilon")] public float clipFactor = 0.2f;                                                                                                                 
        [Range(1e-4f, 1e-2f), Tooltip("beta")] public float entropyRegularization = 0.001f;                                                                                                     
                                                                                                                                                                                                
        [Header("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")]
        [SerializeField] private BufferSize bufferSize = BufferSize.size256;    
        [SerializeField] private BatchSize batchSize = BatchSize.size32;
        [Min(1)] public int epochs = 32;
        public bool normalizeObservations = true;
        public bool normalizeAdvantages = true;

        [HideInInspector] public int buffer_size;
        [HideInInspector] public int batch_size;
        private void Awake()
        {
            buffer_size = (int)Math.Pow(2, (int)bufferSize + 6);
            batch_size = (int)Math.Pow(2, (int)batchSize + 4);
            if (batch_size > buffer_size)
            {
                batch_size = buffer_size;
                batchSize = (BatchSize)(int)bufferSize + 2;
            }
        }
    }
    public enum BufferSize
    {
        size64,
        size128,
        size256,
        size512,
        size1024,
        size2048,
        size4096,
        size8192,
    }
    public enum BatchSize
    {
        size16,
        size32,
        size64,
        size128,
        size256,
        size512,
        size1024,
        size2048,
    }


























    [CustomEditor(typeof(PPOHyperParameters), true), CanEditMultipleObjects]
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