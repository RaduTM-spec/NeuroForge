using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    public class InnovationHistory
    {
        public static InnovationHistory Instance;

        Dictionary<(int, int), int> history; // <(in_id,out_id), innovation>
        int globalLastInnovation = 0;

        public InnovationHistory(Genome startingGenome)
        {
            history = new Dictionary<(int, int), int>();

            if (startingGenome == null)
                return;
           
            // initialize the history based on the starting genome
            foreach (var conn in startingGenome.connections)
            {
                history.Add(                  
                    (conn.Value.inNeuron, conn.Value.outNeuron),
                    conn.Value.innovation
                    );
            }
            globalLastInnovation = startingGenome.GetLastInnovation();         
        }

        public int GetInnovationNumber(int from, int to)
        {
            // check if this connection has ever been seen before
            var key = (from, to);
            if(history.ContainsKey(key))
            {
                // the connection already exists
                return history[key];
            }
            else
            {
                // the connection is novel
                globalLastInnovation++;
                history.Add((from, to), globalLastInnovation);
                return globalLastInnovation;
            }

        }
    }
}