using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuroForge
{
    /// <summary>
    /// This class collects every sensor script that belongs to a gameObject (Agent) and 
    /// collects their values each time CollectObservation() method is called.
    /// </summary>
    internal class AgentSensor
    {
        List<object> sensors;
        public AgentSensor(Transform agent)
        {
            sensors = new List<object>();
            InitSensors(agent);
        }
        public void CollectObservations(SensorBuffer buffer)
        {
            foreach (var item in sensors)
            {
                if (item.GetType() == typeof(RaySensor))
                {
                    RaySensor sens = (RaySensor)item;
                    buffer.AddObservation(sens.GetObservations());
                }
                else
                if (item.GetType() == typeof(CameraSensor))
                {
                    CameraSensor sens = (CameraSensor)item;
                    buffer.AddObservation(sens.FlatCapture());
                }
            }
        }

        private void InitSensors(Transform parent)
        {
            RaySensor rayFound = parent.GetComponent<RaySensor>();
            CameraSensor camFound = parent.GetComponent<CameraSensor>();

            if (rayFound != null && rayFound.enabled)
                sensors.Add(rayFound);
            if (camFound != null && camFound.enabled)
                sensors.Add(camFound);

            foreach (Transform child in parent)
            {
                InitSensors(child);
            }
        }
    }
}

