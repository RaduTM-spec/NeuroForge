using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace SmartAgents
{
    [System.Serializable]
    public class ExperienceBuffer : ScriptableObject, IClearable
    {
        [SerializeField] public List<Sample> records;
        public ExperienceBuffer(string name, bool createScriptableObject)
        {
            records = new List<Sample>();

            if (name == null)
                name = "NewExperienceBuffer";
            name += "#" + UnityEngine.Random.Range(1, 1000) + ".asset";

            if (!createScriptableObject)
                return;
            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name);
            AssetDatabase.SaveAssets();
        }

        public void Store(double[] state, double[] action, double reward, double[] log_probs,double value, bool isEpisodeEnd)
        {
            records.Add(new Sample(state, action, reward, log_probs, value, isEpisodeEnd));
        }
        public bool IsFull(int capacity)
        {
            return records.Count >= capacity;
        }
        public void Clear()
        {
            records.Clear();
        }
        public string ToString()
        {
            return "Experience buffer [" + records.Count + "] type (state,action,reward,advantage)";
        }
    }
 
}