using System;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;

namespace NeuroForge
{
    public class SensorBuffer : IClearable
    {
        public double[] observations;
        private int counter;
        public SensorBuffer(int capacity)
        {
            observations = new double[capacity];
            for (int i = 0; i < capacity; i++)
                observations[i] = 0;
            counter = 0;
        }        
        public void Clear()
        {
            observations = Enumerable.Repeat(0.0, observations.Length).ToArray();
            counter = 0;
        }
        public int GetBufferCapacity()
        {
            if (observations == null)
                return 0;
            else return observations.Length;
        }

        /// <summary>
        /// Appends a bool value to the SensorBuffer. (it is converted into 0 for false and 1 for true)
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(bool observation1)
        {
            if (counter == observations.Length)
            {
                Debug.LogError("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            observations[counter++] = observation1 ? 1 : 0;
        }
        /// <summary>
        /// Appends a float value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(float observation1)
        {
            if (counter == observations.Length)
            {
                Debug.LogError("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            observations[counter++] = observation1;
        }
        /// <summary>
        /// Appends a double value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(double observation1)
        {
            if (counter == observations.Length)
            {
                Debug.LogError("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            observations[counter++] = observation1;
        }
        /// <summary>
        ///  Appends an int value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(int observation1)
        {
            if (counter == observations.Length)
            {
                Debug.LogError("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            observations[counter++] = observation1;
        }
        /// <summary>
        /// Appends an unsigned int value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(uint observation1)
        {
            if (counter == observations.Length)
            {
                Debug.LogError("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            observations[counter++] = observation1;
        }
        /// <summary>
        /// Appends a Vector2 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation2">Value of the observation</param>
        public void AddObservation(Vector2 observation2)
        {
            if (observations.Length - counter < 2)
            {
                Debug.LogError("SensorBuffer available space is " + (observations.Length - counter) + ". Vector2 observation of size 2 is too large.");
                return;
            }
            observations[counter++] = observation2.x;
            observations[counter++] = observation2.y;
        }
        /// <summary>
        /// Appends a Vector3 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation3">Value of the observation</param>
        public void AddObservation(Vector3 observation3)
        {
            if (observations.Length - counter < 3)
            {
                Debug.LogError("SensorBuffer available space is " + (observations.Length - counter) + ". Vector3 observation of size 3 is too large.");
                return;
            }
            observations[counter++] = observation3.x;
            observations[counter++] = observation3.y;
            observations[counter++] = observation3.z;
        }
        /// <summary>
        /// Appends a Vector4 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation4">Value of the observation</param>
        public void AddObservation(Vector4 observation4)
        {

            if (observations.Length - counter < 4)
            {
                Debug.LogError("SensorBuffer available space is " + (observations.Length - counter) + ". Vector4 observation of size 4 is too large.");
                return;
            }

            observations[counter++] = observation4.x;
            observations[counter++] = observation4.y;
            observations[counter++] = observation4.z;
            observations[counter++] = observation4.w;
        }
        /// <summary>
        /// Appends a Quaternion values to the SensorBuffer.
        /// </summary>
        /// <param name="observation4">Value of the observation</param>
        public void AddObservation(Quaternion observation4)
        {
            if (observations.Length - counter < 4)
            {
                Debug.LogError("SensorBuffer available space is " + (observations.Length - counter) + ". Quaternion observation of size 4 is too large.");
                return;
            }
            observations[counter++] = observation4.x;
            observations[counter++] = observation4.y;
            observations[counter++] = observation4.z;
            observations[counter++] = observation4.w;
        }
        /// <summary>
        /// Appends a Transform values to the SensorBuffer.
        /// </summary>
        /// <param name="observation10">Value of the observation</param>
        public void AddObservation(Transform obsevation10)
        {
            if (observations.Length - counter < 10)
            {
                Debug.LogError("SensorBuffer available space is " + (observations.Length - counter) + ". Transform observation of size 10 is too large.");
                return;
            }
            AddObservation(obsevation10.position);
            AddObservation(obsevation10.localScale);
            AddObservation(obsevation10.rotation);
        }
        /// <summary>
        /// Appends an array of double values to the SensorBuffer.
        /// </summary>
        /// <param name="observations">Values of the observations</param>
        public void AddObservation(double[] observations)
        {
            if (this.observations.Length - counter < observations.Length)
            {
                Debug.LogError("SensorBuffer available space is " + (this.observations.Length - counter) + ". Double array observations is too large.");
                return;
            }
            foreach (var item in observations)
            {
                AddObservation(item);
            }
        }
        /// <summary>
        /// Appends an array of float values to the SensorBuffer.
        /// </summary>
        /// <param name="observations">Values of the observations</param>
        public void AddObservation(float[] observations)
        {
            if (this.observations.Length - counter < observations.Length)
            {
                Debug.LogError("SensorBuffer available space is " + (this.observations.Length - counter) + ". Float array observations size is " + observations.Length + ".");
                return;
            }
            foreach (var item in observations)
            {
                AddObservation(item);
            }
        }
        /// <summary>
        /// Appends the distances/infos of each RayCast by the RaySensor to SensorBuffer.
        /// </summary>
        /// <param name="raySensor">RaySensor object</param>
        public void AddObservation(RaySensor raySensor)
        {
            if (raySensor == null)
            {
                Debug.LogError("<color=red>RaySensor added as an observation is null!.</color>");
                return;
            }
            if (this.observations.Length - counter < raySensor.GetObservationSize())
            {
                Debug.LogError("SensorBuffer available space is " + (this.observations.Length - counter) + ". Sensor's observations array is too large.");
                return;
            }
            AddObservation(raySensor.GetObservations());
        }

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append("[ ");

            foreach (var obs in observations)
            {
                stringBuilder.Append(obs);
                stringBuilder.Append(", ");
            }
            stringBuilder.Remove(stringBuilder.Length - 2, 1);
            stringBuilder.Append("]");
            return stringBuilder.ToString();
        }
    }
    

}