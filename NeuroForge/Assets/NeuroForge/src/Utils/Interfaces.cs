using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    public interface IResetable
    {
        void Reset();
    }
    public interface IClearable
    {
        public void Clear();
    }
}
