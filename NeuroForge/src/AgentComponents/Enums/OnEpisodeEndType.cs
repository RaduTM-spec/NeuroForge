using UnityEngine;

namespace NeuroForge
{
    public enum OnEpisodeEndType
    {
        [Tooltip("Neither agent nor environment resets at the end of the episode")]
        ResetNone,
        [Tooltip("Only agent position resets at the end of the episode")]
        ResetAgentOnly,
        [Tooltip("Both agent and the environment reset at the end of the episode")]
        ResetEnvironment
    }
}
