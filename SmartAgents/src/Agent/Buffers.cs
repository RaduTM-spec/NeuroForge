using UnityEngine;
using DRLAgents;

namespace SmartAgents
{
    public class SensorBuffer
    {
        private float[] buffer;
        private int sizeIndex;
        public SensorBuffer(int capacity)
        {
            buffer = new float[capacity];
            for (int i = 0; i < capacity; i++)
                buffer[i] = 0;
            sizeIndex = 0;
        }
        /// <summary>
        /// Returns the array that contains all the input values .
        /// </summary>
        /// <returns>float[] with all values</returns>
        public float[] GetBuffer()
        {
            return buffer;
        }
        public int GetBufferCapacity()
        {
            if (buffer == null)
                return 0;
            else return buffer.Length;
        }


        /// <summary>
        /// Appends a float value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(float observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        /// <summary>
        ///  Appends an int value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(int observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        /// <summary>
        /// Appends an unsigned int value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(uint observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        /// <summary>
        /// Appends a Vector2 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation2">Value of the observation</param>
        public void AddObservation(Vector2 observation2)
        {
            if (buffer.Length - sizeIndex < 2)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector2 observation of size 2 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation2.x;
            buffer[sizeIndex++] = observation2.y;
        }
        /// <summary>
        /// Appends a Vector3 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation3">Value of the observation</param>
        public void AddObservation(Vector3 observation3)
        {

            if (buffer.Length - sizeIndex < 3)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector3 observation of size 3 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation3.x;
            buffer[sizeIndex++] = observation3.y;
            buffer[sizeIndex++] = observation3.z;
        }
        /// <summary>
        /// Appends a Vector4 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation4">Value of the observation</param>
        public void AddObservation(Vector4 observation4)
        {

            if (buffer.Length - sizeIndex < 4)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector4 observation of size 4 is too large.");
                return;
            }

            buffer[sizeIndex++] = observation4.x;
            buffer[sizeIndex++] = observation4.y;
            buffer[sizeIndex++] = observation4.z;
            buffer[sizeIndex++] = observation4.w;
        }
        /// <summary>
        /// Appends a Quaternion value to the SensorBuffer.
        /// </summary>
        /// <param name="observation4">Value of the observation</param>
        public void AddObservation(Quaternion observation4)
        {
            if (buffer.Length - sizeIndex < 4)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Quaternion observation of size 4 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation4.x;
            buffer[sizeIndex++] = observation4.y;
            buffer[sizeIndex++] = observation4.z;
            buffer[sizeIndex++] = observation4.w;
        }
        /// <summary>
        /// Appends a Transform value to the SensorBuffer.
        /// </summary>
        /// <param name="observation10">Value of the observation</param>
        public void AddObservation(UnityEngine.Transform obsevation10)
        {
            if (buffer.Length - sizeIndex < 10)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Transform observation of size 10 is too large.");
                return;
            }
            AddObservation(obsevation10.position);
            AddObservation(obsevation10.localScale);
            AddObservation(obsevation10.rotation);
        }
        /// <summary>
        /// Appends an array of float values to the SensorBuffer.
        /// </summary>
        /// <param name="observations">Values of the observations</param>
        public void AddObservation(float[] observations)
        {
            if (buffer.Length - sizeIndex < observations.Length)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Float array observations is too large.");
                return;
            }
            foreach (var item in observations)
            {
                AddObservation(item);
            }
        }
        /// <summary>
        /// Appends the distances of each RayCast by the RaySensor to SensorBuffer.
        /// </summary>
        /// <param name="raySensor">RaySensor object</param>
        public void AddObservation(RaySensor raySensor)
        {
            if (raySensor == null)
            {
                Debug.LogError("<color=red>RaySensor added as an observation is null!.</color>");
                return;
            }
            AddObservation(raySensor.observations);
        }
    }
    public class ActionBuffer
    {
        private float[] buffer;
        public ActionBuffer(float[] actions)
        {
            buffer = actions;
        }
        public ActionBuffer(int capacity)
        {
            buffer = new float[capacity];
        }

        /// <summary>
        /// Get the buffer array with every action values.
        /// <para>Can be used instead of using GetAction() method.</para>
        /// </summary>
        /// <returns>float[] copy of the buffer</returns>
        public float[] GetBuffer()
        {
            return buffer;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>Total actions number</returns>
        public int GetBufferCapacity()
        {
            return buffer != null ? buffer.Length : 0;
        }
        /// <summary>
        /// Returns the value from the index parameter.
        /// </summary>
        /// <param name="index">The index of the action from ActionBuffer.</param>
        /// <returns>float</returns>
        public float GetAction(uint index)
        {
            try
            {
                return buffer[index];
            }
            catch { Debug.LogError("Action index out of range."); }
            return 0;
        }
        /// <summary>
        /// Sets the action from ActionBuffer with a specific value.
        /// </summary>
        /// <param name="index">The index of the action from ActionBuffer</param>
        /// <param name="action1">The value of the action to be set</param>
        public void SetAction(uint index, float action1)
        {
            buffer[index] = action1;
        }
        /// <summary>
        /// Returns the index of the max value from ActionBuffer.
        /// <para>Usually used when SoftMax is the output activation function.</para>
        /// </summary>
        /// <returns>The index or -1 if all elements are equal.</returns>
        public int Predict()
        {
            float max = float.MinValue;
            int index = -1;
            bool equal = true;
            for (int i = 0; i < buffer.Length; i++)
            {
                if (i > 0 && buffer[i] != buffer[i - 1])
                    equal = false;

                if (buffer[i] > max)
                {
                    max = buffer[i];
                    index = i;
                }
            }
            return equal == true ? -1 : index;

        }
    }

}