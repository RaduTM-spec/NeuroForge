using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [System.Serializable]
    public class PPOMemory : ScriptableObject, IClearable
    {
        [SerializeField] public List<PPOSample> records;
        public PPOMemory(bool createScriptableObject)
        {
            records = new List<PPOSample>();

            if (!createScriptableObject)
                return;

            string name = GenerateName();
            AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
            AssetDatabase.SaveAssets();
            Debug.Log(name + " was created!");
        }

        public void Store(double[] observations, double[] outputs, double reward, double[] log_probs,double value, bool isEpisodeEnd) 
                         => records.Add(new PPOSample(observations, outputs, reward, log_probs, value, isEpisodeEnd));
        public void Clear() => records.Clear();
        public bool IsFull(int capacity) => records.Count >= capacity;   
        public override string ToString() => "Experience buffer [" + records.Count + "]";
        string GenerateName()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<NeuralNetwork>("Assets/BufferXP#" + id + ".asset") != null)
                id++;
            return "BufferXP#" + id;
        }
    }
 
}