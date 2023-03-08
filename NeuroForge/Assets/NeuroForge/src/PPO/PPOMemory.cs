using NeuroForge;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
        }

        public void Store(double[] observations, double[] outputs, double reward, double[] log_probs,double value, bool isEpisodeEnd) 
                         => records.Add(new PPOSample(observations, outputs, reward, log_probs, value, isEpisodeEnd));
        public void Clear() => records.Clear();
        public bool IsFull(int capacity) => records.Count >= capacity;   
        public override string ToString()
        {
            StringBuilder stringBuilder= new StringBuilder();
            foreach (var item in records)
            {
                stringBuilder.Append("[ value: " + item.value + " ] ");
                stringBuilder.Append("[ reward: " + item.reward.ToString("0.0") + " ] ");
                stringBuilder.Append("[ done: " + item.done + " ]\n");
            }
            return stringBuilder.ToString();
        }
        string GenerateName()
        {
            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<PPOMemory>("Assets/Memory#" + id + ".asset") != null)
                id++;
            return "Memory#" + id;
        }
    }
 
}