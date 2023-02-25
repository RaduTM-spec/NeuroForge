using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [DisallowMultipleComponent, AddComponentMenu("NeuroForge/HyperParameters")]
    public class NEATHyperParameters : MonoBehaviour 
    {
        [Header("Session")]
        [Min(50)] public int generations = 1000;
        [Min(5), Tooltip("seconds")] public int timeHorizon = 60;

        [Header("Individuals")]
        [Min(1)] public int populationSize = 100;      
        [Range(.2f, .8f)] public float survivalRate = .5f;

        [Header("Speciation")]
        [Min(0)] public float delta = 3f;
        [Min(0)] public float c1 = 1f;
        [Min(0)] public float c2 = 1f;
        [Min(0)] public float c3 = 0.4f;

        [Header("Mutation")] 
        [Range(0, 1)] public float addConnection = 0.05f;
        [Range(0, 1)] public float addNode = 0.01f;
        [Range(0,1)] public float mutateConnections = 0.80f;
        [Range(0, 1)] public float mutateNode = 0.04f;
        [Range(0,1)] public float noMutation = 0.10f;

        [Header("Genome")]
        [Min(30)] public int maxConnections = 100;
        [Min(5)] public int maxNodes = 20;
    }

    [CustomEditor(typeof(NEATHyperParameters), true), CanEditMultipleObjects]
    class ScriptlessNEATHP : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawPropertiesExcluding(serializedObject, new string[] { "m_Script" });

            // Applied SoftMax on probabilities
            SerializedProperty addCon = serializedObject.FindProperty("addConnection");
            SerializedProperty mutCon = serializedObject.FindProperty("mutateConnections");
            SerializedProperty addNod = serializedObject.FindProperty("addNode");
            SerializedProperty mutNod = serializedObject.FindProperty("mutateNode");
            SerializedProperty nonMut = serializedObject.FindProperty("noMutation");

            float exp_sum = 1e-8f;
            exp_sum += addCon.floatValue;
            exp_sum += mutCon.floatValue;
            exp_sum += addNod.floatValue;
            exp_sum += mutNod.floatValue;
            exp_sum += nonMut.floatValue;

            addCon.floatValue /= exp_sum;
            mutCon.floatValue /= exp_sum;
            addNod.floatValue /= exp_sum;
            mutNod.floatValue /= exp_sum;
            nonMut.floatValue /= exp_sum;

            addCon.floatValue = (float)Math.Round((double)addCon.floatValue, 2);
            mutCon.floatValue = (float)Math.Round((double)mutCon.floatValue, 2);
            addNod.floatValue = (float)Math.Round((double)addNod.floatValue, 2);
            mutNod.floatValue = (float)Math.Round((double)mutNod.floatValue, 2);
            nonMut.floatValue = (float)Math.Round((double)nonMut.floatValue, 2);

            serializedObject.ApplyModifiedProperties();
        }
    }
}
