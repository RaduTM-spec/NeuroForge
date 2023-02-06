using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    public class InnovationCounter
    {
        public static int counter = 1;

        public InnovationCounter(int start = 1)
        {
            counter = start;
        }
        public int GetInnovation() => counter++;
    }
}