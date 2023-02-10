using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [DisallowMultipleComponent, AddComponentMenu("NeuroForge/HyperParameters")]
    public class NEATHyperParameters : MonoBehaviour 
    {
        [Min(1)]public int populationSize = 50;
        [Min(5), Tooltip("seconds")] public int maxEpsiodeLength = 60;

        [Header("Compatibility")]
        public float delta = 3f;
        public float c1 = 1f;
        public float c2 = 1f;
        public float c3 = 0.4f;

        [Header("Mutation")]
        [Range(0,1)] public float addConnection = 0.10f;
        [Range(0,1)] public float removeConnection = 0.15f;
        [Range(0,1)] public float mergeConnections = 0.20f;
        [Range(0,1)] public float mutateConnections = 0.20f;
        [Range(0,1)] public float addNode = 0.10f;
        [Range(0,1)] public float mutateNode = 0.15f;
        [Range(0,1)] public float noMutation = 0.10f;
        
        

    }

    [CustomEditor(typeof(NEATHyperParameters), true), CanEditMultipleObjects]
    class ScriptlessNEATHP : Editor
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
