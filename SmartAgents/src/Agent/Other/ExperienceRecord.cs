using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[System.Serializable]
public class ExperienceRecord : ScriptableObject
{
    public List<Sample> records;
    public ExperienceRecord(string name = null) { 
         records= new List<Sample>();
    
         if (name == null)
             name = "NewER";
         name += "#" + UnityEngine.Random.Range(1, 1000) + ".asset";
    
         Debug.Log(name + " was created!");
         AssetDatabase.CreateAsset(this, "Assets/" + name);
         AssetDatabase.SaveAssets();
         EditorGUIUtility.SetIconForObject(this, AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/SmartAgents/doc/experience_records_icon.png"));
    }

    public void AddRecord(Sample sample)
    {
        records.Add(sample);
    }
    public void AddRecord(double[] state, double[] actions, double reward, double[] nextState)
    {
        records.Add(new Sample(state, actions, reward, nextState));
    }


}
