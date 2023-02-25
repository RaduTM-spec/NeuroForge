using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    [DisallowMultipleComponent, AddComponentMenu("NeuroForge/HyperParameters")]
    public class NEATHyperParameters : MonoBehaviour 
    {
        [Header("Session")]
        [Min(50)] public int generations = 1000;
        [Min(5), Tooltip("seconds")] public int timeHorizon = 60;

        [Header("Individuals")]
        [Min(1)] public int populationSize = 100;
        [Min(5), Tooltip("generations")] public int stagnationAllowance = 15;
        [Range(.2f, .8f)] public float survivalRate = .5f;
        

        [Header("Speciation")]
        [Min(0)] public float delta = 3f;
        [Min(0)] public float c1 = 1f;
        [Min(0)] public float c2 = 1f;
        [Min(0)] public float c3 = 0.4f;

        [Header("Mutation")]
        [Min(0.01f)] public float weightDeviationStrength = 0.05f;

        [Range(0, 1)] public float addConnection = 0.05f;
        [Range(0, 1)] public float addNode = 0.01f;
        [Range(0,1)] public float mutateConnections = 0.80f;
        [Range(0, 1)] public float mutateNode = 0.04f;
        [Range(0,1)] public float noMutation = 0.10f;

        [Header("Genome")]
        [Min(30)] public int maxConnections = 100;
        [Min(5)] public int maxNodes = 20;
    }

    [CustomEditor(typeof(NEATHyperParameters), true), CanEditMultipleObjects]
    class ScriptlessNEATHP : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawPropertiesExcluding(serializedObject, new string[] { "m_Script" });

            // Applied SoftMax on probabilities
            SerializedProperty addCon = serializedObject.FindProperty("addConnection");
            SerializedProperty mutCon = serializedObject.FindProperty("mutateConnections");
            SerializedProperty addNod = serializedObject.FindProperty("addNode");
            SerializedProperty mutNod = serializedObject.FindProperty("mutateNode");
            SerializedProperty nonMut = serializedObject.FindProperty("noMutation");

            float exp_sum = 1e-8f;
            exp_sum += addCon.floatValue;
            exp_sum += mutCon.floatValue;
            exp_sum += addNod.floatValue;
            exp_sum += mutNod.floatValue;
            exp_sum += nonMut.floatValue;

            addCon.floatValue /= exp_sum;
            mutCon.floatValue /= exp_sum;
            addNod.floatValue /= exp_sum;
            mutNod.floatValue /= exp_sum;
            nonMut.floatValue /= exp_sum;

            addCon.floatValue = (float)Math.Round((double)addCon.floatValue, 2);
            mutCon.floatValue = (float)Math.Round((double)mutCon.floatValue, 2);
            addNod.floatValue = (float)Math.Round((double)addNod.floatValue, 2);
            mutNod.floatValue = (float)Math.Round((double)mutNod.floatValue, 2);
            nonMut.floatValue = (float)Math.Round((double)nonMut.floatValue, 2);

            serializedObject.ApplyModifiedProperties();
        }
    }

        // Original Paper -> Parameter settings
         /*The same experimental settings are used in all experiments; they were not tuned specifically
        for any particular problem. The one exception is the hardest pole balancing problem
        (Double pole, no velocities, or DPNV) where a larger population size was used to
        match those of other systems in this task. Because some of NEAT’s system parameters
        are sensitive to population size, we altered them accordingly.
        All experiments except DPNV, which had a population of 1,000, used a population
        of 150 NEAT networks. The coefficients for measuring compatibility were c1 = 1:0,
        c2 = 1:0, and c3 = 0:4. With DPNV, c3 was increased to 3:0 in order to allow for finer
        distinctions between species based on weight differences (the larger population has
        room for more species). In all experiments, t = 3:0, except in DPNV where it was 4:0,
        to make room for the larger weight significance coefficient c3. If the maximum fitness of
        a species did not improve in 15 generations, the networks in the stagnant species were
        not allowed to reproduce. The champion of each species with more than five networks
        was copied into the next generation unchanged. There was an 80% chance of a genome
        having its connection weights mutated, in which case each weight had a 90% chance of
        being uniformly perturbed and a 10% chance of being assigned a new random value.
        (The system is tolerant to frequent mutations because of the protection speciation provides.) 
         
        There was a 75% chance that an inherited gene was disabled if it was disabled
        in either parent. In each generation, 25% of offspring resulted from mutation without
        crossover. The interspecies mating rate was 0.001. In smaller populations, the probability
        of adding a new node was 0.03 and the probability of a new link mutation was
        0.05. In the larger population, the probability of adding a new link was 0.3, because a
        larger population can tolerate a larger number of prospective species and greater topological
        diversity. We used a modified sigmoidal transfer function, '(x) = 1
        1+e4:9x , at
        all nodes. The steepened sigmoid allows more fine tuning at extreme activations. It
        is optimized to be close to linear during its steepest ascent between activations 0:5
        and 0:5. These parameter values were found experimentally: links need to be added
        significantly more often than nodes, and an average weight difference of 3.0 is about as
        significant as one disjoint or excess gene. Performance is robust to moderate variations
        in these values.*/
}
