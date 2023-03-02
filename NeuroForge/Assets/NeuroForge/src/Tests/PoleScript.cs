using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoleScript : MonoBehaviour
{
    public PoleBalancer poleBalancer;
    public void EndEpisodeOfBalancer() => poleBalancer.EndEpisode();
}
