using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    /// <summary>
    /// The environment is public foreach agent, they are overlayed
    /// TODOs: change const values from network mutation to acces TRAINER hyper parameters (or mayb let mutate on their own ........don t really do the change honstly_)
    /// </summary>
    public sealed class NEATTrainer : MonoBehaviour
    {
        private static NEATTrainer Instance;

        [SerializeField] private List<NEATAgent> agents;

        private NEATNetwork mainModel;
        private NEATHyperParameters hp;
        private TrainingEnvironment trainingEnvironment;

        private int agentsDead = 0;
        private int episodeLength = 60;
        private float episodeTimePassed = 0;


        private InnovationCounter innovationCounter;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
            }
        }
        private void Update()
        {
            if(Instance)
                Instance.episodeTimePassed += Time.deltaTime;
        }
        private void LateUpdate()
        {   
            if(Instance != null && (Instance.episodeTimePassed >= Instance.episodeLength || Instance.agentsDead == Instance.agents.Count))
            {
                // Update NEAT



                // Revive all agents
                foreach (var agent in Instance.agents)
                {
                    agent.Ressurect();
                }
            }
        }

        public static void Initialize(NEATAgent agent)
        {
            if(Instance == null)
            {
                GameObject go = new GameObject("PPOTrainer");
                go.AddComponent<NEATTrainer>();
                Instance.agents = new List<NEATAgent>();
                Instance.mainModel = agent.model;
                Instance.hp = agent.hp;
                Instance.episodeLength = agent.maxEpsiodeLength;
                Instance.innovationCounter = new InnovationCounter(agent.model.GetNewestWeightInnovation() + 1);
                // Init environment
                if (agent.GetOnEpisodeEndType() == OnEpisodeEndType.ResetNone) return;
                Instance.trainingEnvironment = agent.GetOnEpisodeEndType() == OnEpisodeEndType.ResetEnvironment ?
                                    Instance.trainingEnvironment = new TrainingEnvironment(agent.transform.parent) :
                                    Instance.trainingEnvironment = new TrainingEnvironment(agent.transform);
            }

            Instance.agents.Add(agent);
        }
        public static void Ready()
        {
            if(Instance)
            Instance.agentsDead++;
        }


        public static NEATNetwork CrossOver(NEATNetwork parent1, NEATNetwork parent2)
        {
            NEATNetwork child = new NEATNetwork(parent1.GetInputsNumber(), parent1.outputShape, parent1.actionSpace, false);

            int lastInov = parent1.GetNewestWeightInnovation() > parent2.GetNewestWeightInnovation() ? 
                           parent1.GetNewestWeightInnovation() : parent2.GetNewestWeightInnovation();

            for (int gene = 0; gene < lastInov; gene++)
            {
                ConnectionGene parent1_gene;
                if (parent1.connections.TryGetValue(gene, out parent1_gene)) {; }
                ConnectionGene parent2_gene;
                if (parent1.connections.TryGetValue(gene, out parent2_gene)) {; }

                if (parent1_gene == null && parent2_gene == null) continue;
                if (parent1_gene == null) AssignGene(parent2_gene);
                else if (parent2_gene == null) AssignGene(parent1_gene);
                else if (Random.value < .5f) AssignGene(parent1_gene);
                else AssignGene(parent2_gene);

                void AssignGene(ConnectionGene g)
                {
                    int node1 = g.inNeuron;
                    int node2 = g.outNeuron;
                    int inovation = g.innovation;

                    NodeGene inNode = child.nodes[node1];
                    NodeGene outNode = child.nodes[node2];

                    // Also add the nodes in the child genome
                    if(inNode == null)
                    {
                        NodeGene toAdd = new NodeGene(node1, NEATNodeType.hidden);
                        child.nodes.Add(toAdd.innovation, toAdd);
                        inNode = toAdd;
                    }
                    if(outNode == null)
                    {
                        NodeGene toAdd = new NodeGene(node2, NEATNodeType.hidden);
                        child.nodes.Add(toAdd.innovation, toAdd);
                        inNode = toAdd;
                    }

                    ConnectionGene crossedGene = new ConnectionGene(inNode, outNode, inovation);
                    child.connections.Add(inovation, crossedGene);
                }


            }


            return child;
           
        }

        public static float Distance(NEATNetwork parent1, NEATNetwork parent2)
        {
            float c1 = Instance.hp.c1;
            float c2 = Instance.hp.c2;
            float c3 = Instance.hp.c3;

            float N = parent1.connections.Count > parent2.connections.Count ? parent1.connections.Count : parent2.connections.Count;
            float E = 0;
            float D = 0;
            float W = 0;
            int wCount = 0;

            int lastInov = parent1.GetNewestWeightInnovation() > parent2.GetNewestWeightInnovation() ?
                           parent1.GetNewestWeightInnovation() : parent2.GetNewestWeightInnovation();

            // Calculate D and W
            for (int i = 0; i < lastInov; i++)
            {
                ConnectionGene parent1_gene;
                if (parent1.connections.TryGetValue(i, out parent1_gene)) {; }
                ConnectionGene parent2_gene;
                if (parent1.connections.TryGetValue(i, out parent2_gene)) {; }

                if (parent1_gene == null && parent2_gene == null) continue;
                if (parent1_gene != null && parent2_gene != null)
                {
                    wCount++;
                    W += Mathf.Abs(parent1_gene.weight - parent2_gene.weight);
                }
                else D++;
            }
            // Calculate E
            for (int i = lastInov - 1; i >= 0; i--)
            {

                ConnectionGene parent1_gene;
                if (parent1.connections.TryGetValue(i, out parent1_gene)) { }
                ConnectionGene parent2_gene;
                if (parent1.connections.TryGetValue(i, out parent2_gene)) { }

                if (parent1_gene != null && parent2_gene != null) break;
                E++;
            }

            D -= E;
            W /= wCount;

            Debug.Log("E = " + E + " | D = " + D + " | W = " + W + " | N = " + N );
            return (c1 * E / N) + (c2 * D / N) + (c3 * W);
        }










        public static void Dispose() => Instance = null;
        public static int GetInnovation() => Instance.innovationCounter.GetInnovation();
        public static void InitializeHyperParameters() => Instance.hp = new NEATHyperParameters();
    }
}

