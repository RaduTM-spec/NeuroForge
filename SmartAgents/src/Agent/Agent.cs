using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace SmartAgents
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        public BehaviorType behavior = BehaviorType.Passive;
        [SerializeField] private ArtificialNeuralNetwork Network;

        [Space]
        [Min(1), SerializeField] private int SpaceSize = 2;
        [Min(1), SerializeField] private int ActionSize = 2;
        [SerializeField] private ActionType actionType = ActionType.Continuous;

        [Space]
        [SerializeField] private bool UseRaySensors = true;
        

        private HyperParameters hyperParamters;
        private List<RaySensor> raySensors = new List<RaySensor>();

        private SensorBuffer sensorBuffer;
        private ActionBuffer actionBuffer;

        protected virtual void Awake()
        {
            hyperParamters = GetComponent<HyperParameters>();
            InitNetwork();
            InitBuffers();
            InitRaySensors(this.gameObject);  
        }
        void InitNetwork()
        {
            if (Network != null)
                return;

            ActivationType activation = hyperParamters.activationType;
            ActivationType outputActivation = hyperParamters.activationType;
            LossType loss = hyperParamters.lossType;
            
            if (actionType == ActionType.Discrete)
            {
                outputActivation = ActivationType.SoftMax;
                loss = LossType.CrossEntropy;
                
            }
            else if (actionType == ActionType.Continuous)
            {
                outputActivation = ActivationType.Tanh;
            }
            
            Network = new ArtificialNeuralNetwork(SpaceSize, ActionSize, hyperParamters.networkHiddenLayers, activation, outputActivation, loss);
            Network.Save();
            
        }
        void InitBuffers()
        {
            sensorBuffer = new SensorBuffer(Network.GetInputsNumber());
            actionBuffer = new ActionBuffer(Network.GetOutputsNumber());
        }
        void InitRaySensors(GameObject parent)
        {
            //adds all 
            if (!UseRaySensors)
                return;

            raySensors.Add(GetComponent<RaySensor>());
            foreach (Transform child in parent.transform)
            {
                InitRaySensors(child.gameObject);
            }
        }    

        protected virtual void Update()
        {
            sensorBuffer.Clear();
            sensorBuffer.Clear();
            switch(behavior)
            {
                case BehaviorType.Active:
                    ActiveAction();
                    break;
                case BehaviorType.Manual:
                    ManualAction();
                    break;
                case BehaviorType.Learn:
                    LearnAction();
                    break;
                case BehaviorType.Heuristic:
                    HeuristicAction();
                    break;
                default:
                    break;

            }
        }

        void ActiveAction()
        {
            if(Network == null)
            {
                Debug.LogError("<color=red>Neural Network is missing. Agent cannot take any actions.</color>");
                return;
            }

            if(UseRaySensors)
            {
                foreach (var raySensor in raySensors)
                {
                    sensorBuffer.AddObservation(raySensor);
                }
            }
            
            CollectObservations(sensorBuffer);
            actionBuffer.actions = Network.ForwardPropagation(sensorBuffer.observations);
            OnActionReceived(actionBuffer);

        }
        void ManualAction()
        {
            Heuristic(actionBuffer);
            OnActionReceived(actionBuffer);
        }
        void LearnAction()
        {

        }
        void HeuristicAction()
        {

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