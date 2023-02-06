using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class PPONetwork : ScriptableObject
    {
        [SerializeField] public PPOActorNetwork actorNetwork;
        [SerializeField] public NeuralNetwork criticNetwork;

        [SerializeField] public RunningNormalizer observationsNormalizer;

        public PPONetwork(int observationsSize, int continuousSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initialization)
        {
            actorNetwork = new PPOActorNetwork(observationsSize, continuousSize, hiddenUnits, layersNum, activation, initialization);
            criticNetwork = new NeuralNetwork(observationsSize, 1, hiddenUnits, layersNum, ActivationType.Tanh, ActivationType.Linear, LossType.MeanSquare, InitializationType.NormalDistribution, true, GetCriticName());
            observationsNormalizer = new RunningNormalizer(observationsSize);
            CreateAsset();
        }
        public PPONetwork(int observationsSize, int[] discreteSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initialization)
        {
            actorNetwork = new PPOActorNetwork(observationsSize, discreteSize, hiddenUnits, layersNum, activation, initialization);
            criticNetwork = new NeuralNetwork(observationsSize, 1, hiddenUnits, layersNum, ActivationType.Tanh, ActivationType.Linear, LossType.MeanSquare, InitializationType.NormalDistribution, true, GetCriticName());
            observationsNormalizer = new RunningNormalizer(observationsSize);
            CreateAsset();
        }
       
        private void CreateAsset()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<PPONetwork>("Assets/PPONetworkNN#" + id + ".asset") != null)
                id++;
            string assetName = "PPONetworkNN#" + id + ".asset";

            AssetDatabase.CreateAsset(this, "Assets/" + assetName);
            AssetDatabase.SaveAssets();
            Debug.Log(assetName + " was created!");
        }
        private string GetCriticName()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<NeuralNetwork>("Assets/CriticNN#" + id + ".asset") != null)
                id++;
            return "CriticNN#" + id;
        }
    }
}
