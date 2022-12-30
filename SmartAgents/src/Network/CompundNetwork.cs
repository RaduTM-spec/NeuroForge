using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace SmartAgents
{
    [Serializable]
    public class CompoundNetwork : ScriptableObject
    {
        [SerializeField] public ArtificialNeuralNetwork actorNetwork;
        [SerializeField] public ArtificialNeuralNetwork criticNetwork;
        [SerializeField] private List<Sample> offlineTrainingData;

        public CompoundNetwork(int inputs, int outputs, HiddenLayers size, ActivationType activationFunction, ActivationType outputActivationFunction, LossType lossFunction, string? name = null) 
        {
            actorNetwork = new ArtificialNeuralNetwork(inputs, outputs, size, activationFunction, outputActivationFunction, lossFunction, true, "Actor");
            criticNetwork = new ArtificialNeuralNetwork((inputs + outputs), 1, size, ActivationType.Tanh, ActivationType.Tanh, LossType.MeanSquare, true, "Critic");
            offlineTrainingData= new List<Sample>();

            if (name == null)
                name = "ActorCriticNetwork#" + UnityEngine.Random.Range(1, 1000);
            Debug.Log(name + " was created!");


            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/compound_network_icon.png"));
        }
        public void AddSample(Sample newSample)
        {
            offlineTrainingData.Add(newSample);
        }
        public int GetActorInputsNumber()
        {
            return actorNetwork.GetInputsNumber();
        }
        public int GetActorOutputsNumber()
        {
            return actorNetwork.GetOutputsNumber();
        }


    }
}