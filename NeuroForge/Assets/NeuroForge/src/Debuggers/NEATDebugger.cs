using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
public class NEATDebugger : MonoBehaviour
{
    public NEATNetwork model;

    private void Start()
    {
        if (model == null)
            model = new NEATNetwork(2, new int[2], ActionType.Continuous, false, true);
        Functions.Print(model.GetContinuousActions(new double[] { 1, 1 }));
        StartCoroutine(Mutation());
    }
    private IEnumerator Mutation()
    {
        yield return new WaitForSeconds(1);
        model.Mutate();
        StartCoroutine(Mutation());
    }
}
