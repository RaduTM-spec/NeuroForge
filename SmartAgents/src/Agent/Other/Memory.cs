using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[System.Serializable]
public class Memory : ScriptableObject
{
    [SerializeField] public List<Sample> records;
    public Memory(string name = null) { 
         records = new List<Sample>();
    
         if (name == null)
             name = "NewMemory";
         name += "#" + UnityEngine.Random.Range(1, 1000) + ".asset";
    
         Debug.Log(name + " was created!");
         AssetDatabase.CreateAsset(this, "Assets/" + name);
         AssetDatabase.SaveAssets();
         EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/memory_icon.png"));
    }

    public void AddRecord(Sample sample)
    {
        records.Add(sample);
    }
    public void PopOldRecord()
    {
        try
        {
            records.RemoveAt(0);
        }
        catch { }
    }

    public bool IsEmpty()
    {
        if(records == null || records.Count == 0) return true;
        return false;
    }
    public int GetSize()
    {
        return records.Count;
    }
}
