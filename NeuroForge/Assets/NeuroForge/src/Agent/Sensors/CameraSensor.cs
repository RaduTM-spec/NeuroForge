using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroForge;
using UnityEditor;
using System.IO;
using UnityEngine.UI;

namespace NeuroForge
{
    [AddComponentMenu("NeuroForge/CameraSensor")]
    public class CameraSensor : MonoBehaviour
    {
        public Camera cam;
        [Min(16)]public int Width = 640;
        [Min(9)]public int Height = 480;
        public ImageType type = ImageType.RGB;
        

        public void Awake()
        {
            if(cam == null)
            {
                Debug.LogError("<color=red>CameraSensor camera not set to an instance of an object.</color>");
                return;
            }
            cam.targetTexture = new RenderTexture(Width, Height, 0);
        }
        public Texture2D Capture()
        {
            RenderTexture activeRT = RenderTexture.active;
            RenderTexture.active = cam.targetTexture;

            cam.Render();

            Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            image.Apply();
            RenderTexture.active = activeRT;

            if (type == ImageType.Greyscale)
                GreyScaleTexture(image);

            return image;
        }
        public float[] FlatCapture()
        {
            Color[] pixels = Capture().GetPixels();

            int i = 0;
            float[] flatPixels;

            if (type == ImageType.RGB)
            {
                flatPixels = new float[pixels.Length * 3];
                foreach (var pixel in pixels)
                {
                    flatPixels[i++] = pixel.r;
                    flatPixels[i++] = pixel.g;
                    flatPixels[i++] = pixel.b;
                }
            }
            else
            {
                flatPixels = new float[pixels.Length];
                foreach (var pixel in pixels)
                    flatPixels[i++] = pixel.grayscale;
            }

            return flatPixels;
        }
        public void TakeShot()
        {
            if (cam == null)
            {
                Debug.LogError("<color=red>CameraSensor camera object reference not set to an instance of an object.</color>");
                return;
            }
            if (cam.targetTexture == null)
            {
                Debug.LogError("<color=red>Camera target texture object reference not set to an instance of an object.</color>");
                return;
            }

            byte[] pngData = Capture().EncodeToPNG();
            Debug.Log(pngData);

            short id = 1;
            while (AssetDatabase.LoadAssetAtPath<Object>("Assets/Shot#" + id + ".png") != null)
                id++;
            string path = "Assets/" + "Shot#" + id + ".png";

            File.WriteAllBytes(path, pngData);
            AssetDatabase.Refresh();
        }

        private void GreyScaleTexture(Texture2D texture)
        {
            Color[] pixels = texture.GetPixels();
            for (int i = 0; i < pixels.Length; i++)
            {
                float greyPixel = pixels[i].grayscale;
                pixels[i] = new Color(greyPixel, greyPixel, greyPixel, pixels[i].a);
            }
            texture.SetPixels(pixels);
        }   
    }

    public enum ImageType
    {
        RGB,
        Greyscale,
    }

    #region Editor
    [CustomEditor(typeof(CameraSensor)), CanEditMultipleObjects]
    class ScriptlessCameraSensor : Editor
    {
        private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };
 
        public override void OnInspectorGUI()
        {
            CameraSensor script = (CameraSensor)target; 

            serializedObject.Update();
            DrawPropertiesExcluding(serializedObject, _dontIncludeMe);
            serializedObject.ApplyModifiedProperties();

            EditorGUILayout.Separator();
            if(GUILayout.Button("Take a shot"))
            {
                script.TakeShot();
            }
        }
    }
    #endregion
}
