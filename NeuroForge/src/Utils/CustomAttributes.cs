using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace NeuroForge
{
    public class NamedArrayAttribute : PropertyAttribute
    {
        public readonly string[] names;
        public NamedArrayAttribute(string[] names) { this.names = names; }
    }


    [CustomPropertyDrawer(typeof(NamedArrayAttribute))]
    public class NamedArrayDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect rect, SerializedProperty property, GUIContent label)
        {
            try
            {
                int pos = int.Parse(property.propertyPath.Split('[', ']')[1]);
                EditorGUI.ObjectField(rect, property, new GUIContent(((NamedArrayAttribute)attribute).names[pos]));
            }
            catch
            {
                EditorGUI.ObjectField(rect, property, label);
            }
        }
    }

}
