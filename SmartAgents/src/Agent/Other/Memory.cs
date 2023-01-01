using SmartAgents;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
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
    public void NormalizeRewards()
    {
        double minReward = records.Min(x => x.reward);
        double maxReward = records.Max(x => x.reward);
        double range = maxReward - minReward;

        for (int i = 0; i < records.Count; i++)
        {
            double normalizedReward;
            if (records[i].reward < 0)
                normalizedReward = -(records[i].reward / minReward);
            else
                normalizedReward = records[i].reward / maxReward;
            Sample unnormalizedSample = records[i];
            unnormalizedSample.reward = normalizedReward;

            records[i] = unnormalizedSample;
        }

    }
    public void CalculateDiscountedRewards(float gamma, ArtificialNeuralNetwork critic)
    {
        for (int i = 0; i < records.Count - 1; i++)
        {
            Sample currentRecord = records[i];
             currentRecord.discountedReward = currentRecord.reward + gamma * DiscountedReward(i+1, gamma, critic);
        }
        double DiscountedReward(int nextRecord, float gamma, ArtificialNeuralNetwork critic)
        {
            if (records[nextRecord].terminalState)
                return records[nextRecord].reward;
            else
            {
                return records[nextRecord].reward + gamma * DiscountedReward(nextRecord+1, gamma, critic);
            }
        }
    }
    


    public void PopLast()
    {
        try
        {
            records.RemoveAt(records.Count - 1); 
        }
        catch { }
    }
    public void PopFirst()
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
