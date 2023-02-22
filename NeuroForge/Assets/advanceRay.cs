using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class advanceRay : MonoBehaviour
{
    public float speed = 5f;
    // Update is called once per frame
    void Update()
    {
        transform.position += new Vector3(0, 0, speed) * Time.deltaTime;
    }
}
