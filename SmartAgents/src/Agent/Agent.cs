using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SmartAgents;
using UnityEditor;
using System.ComponentModel;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        public BehaviorType behavior = BehaviorType.Passive;
        public ArtificialNeuralNetwork Network;

        [Space(10)]
        
        public int SpaceSize = 2;
        [Min(1)] public int ActionSize = 2;
        public HiddenLayers HiddenSize = HiddenLayers.None;
        public ActionType actionType = ActionType.Continuous;

        HyperParameters hyperParamters;
        private void Start()
        {
            hyperParamters = GetComponent<HyperParameters>();
            InitNetwork();

        }
        void InitNetwork()
        {
            if (Network != null)
                return;

            Network = new ArtificialNeuralNetwork(SpaceSize, ActionSize, HiddenSize);
            Network.Save();
        }

        public virtual void CollectObservations(SensorBuffer sensorBuffer)
        {

        }
        public virtual void OnActionReceived(ActionBuffer actionBuffer)
        {

        }
        public virtual void Heuristic(ActionBuffer actionSet) 
        {

        }
    }

    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    class ScriptlessAgent : Editor
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