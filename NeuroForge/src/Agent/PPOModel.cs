using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class PPOModel:ScriptableObject
    {
        [SerializeField] public ActorNetwork actorNetwork;
        [SerializeField] public NeuralNetwork criticNetwork;

        [SerializeField] public OnlineNormalizer observationsNormalizer;
        [SerializeField] public OnlineNormalizer advantagesNormalizer;

        public PPOModel(int obsSize, int continuousSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initialization)
        {
            actorNetwork = new ActorNetwork(obsSize, continuousSize, hiddenUnits, layersNum, activation, initialization);
            criticNetwork = new NeuralNetwork(obsSize, 1, hiddenUnits, layersNum, activation, ActivationType.Linear, LossType.MeanSquare, initialization, true, GetCriticName());
            observationsNormalizer = new OnlineNormalizer(obsSize);
            advantagesNormalizer = new OnlineNormalizer(1);
            CreateAsset();
        }
        public PPOModel(int obsSize, int[] discreteSize, int hiddenUnits, int layersNum, ActivationType activation, InitializationType initialization)
        {
            actorNetwork = new ActorNetwork(obsSize, discreteSize, hiddenUnits, layersNum, activation, initialization);
            criticNetwork = new NeuralNetwork(obsSize, 1, hiddenUnits, layersNum, activation, ActivationType.Linear, LossType.MeanSquare, initialization, true, GetCriticName());
            observationsNormalizer = new OnlineNormalizer(obsSize);
            advantagesNormalizer = new OnlineNormalizer(1);
            CreateAsset();
        }
       
        private void CreateAsset()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<ActorNetwork>("Assets/ModelNN#" + id + ".asset") != null)
                id++;
            string assetName = "ModelNN#" + id + ".asset";

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
