using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace SmartAgents
{
    [System.Serializable]
    public class Memory : ScriptableObject, IClearable
    {
        [SerializeField] public List<Sample> records;
        public Memory(string name = null)
        {
            records = new List<Sample>();

            if (name == null)
                name = "NewMemory";
            name += "#" + UnityEngine.Random.Range(1, 1000) + ".asset";

            Debug.Log(name + " was created!");
            AssetDatabase.CreateAsset(this, "Assets/" + name);
            AssetDatabase.SaveAssets();
            EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/memory_icon.png"));
        }
        public void Store(Sample sample)
        {
            records.Add(sample);
        }
        public void Store(double[] state, double[] action, double reward, bool isEpisodeEnd)
        {
            records.Add(new Sample(state, action, reward, isEpisodeEnd));
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
            return "Memory [" + records.Count + "] type (state,action,reward,advantage)";
        }
    }
    public enum MemorySize
    {
        size256,
        size512,
        size1024,
        size2048,
        size4096,
        size8192,
        size16384,
        size32768,
        size65536
    }
    public enum MiniBatchSize
    {
        size32,
        size64,
        size128,
        size256,
        size512,
        size1024,
        size2048,
        size4096,
        size8192,
        size16384,
    }
}