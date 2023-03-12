using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawShapes : MonoBehaviour
{
    private Vector3 mouseDownPosition;
    private Vector3 mouseUpPosition;
    private bool isDrawing;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            isDrawing = true;
            mouseDownPosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mouseDownPosition.z = 0;
        }

        if (Input.GetMouseButtonUp(0))
        {
            isDrawing = false;
            mouseUpPosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mouseUpPosition.z = 0;
            DrawRectangleOnScreen();
        }
    }

    void DrawRectangleOnScreen()
    {
        Vector3[] corners = new Vector3[4];
        corners[0] = new Vector3(mouseDownPosition.x, mouseDownPosition.y, 0);
        corners[1] = new Vector3(mouseUpPosition.x, mouseDownPosition.y, 0);
        corners[2] = new Vector3(mouseUpPosition.x, mouseUpPosition.y, 0);
        corners[3] = new Vector3(mouseDownPosition.x, mouseUpPosition.y, 0);

        GameObject rectangle = new GameObject();
        rectangle.AddComponent<SpriteRenderer>();
        SpriteRenderer spriteRenderer = rectangle.GetComponent<SpriteRenderer>();
        spriteRenderer.color = Color.blue;
        Texture2D texture = new Texture2D(1, 1);
        texture.SetPixel(0, 0, Color.blue);
        texture.Apply();
        spriteRenderer.sprite = Sprite.Create(texture, new Rect(0, 0, 1, 1), new Vector2(0.5f, 0.5f));
        float width = Mathf.Abs(mouseUpPosition.x - mouseDownPosition.x);
        float height = Mathf.Abs(mouseUpPosition.y - mouseDownPosition.y);
        rectangle.transform.position = new Vector3(mouseDownPosition.x + width / 2, mouseDownPosition.y + height / 2, 0);
        // Set the size of the rectangle
        spriteRenderer.size = new Vector2(width, height);

        // Set the rotation of the rectangle
        float angle = Mathf.Atan2(mouseUpPosition.y - mouseDownPosition.y, mouseUpPosition.x - mouseDownPosition.x) * Mathf.Rad2Deg;
        rectangle.transform.rotation = Quaternion.Euler(new Vector3(0, 0, angle));

        // Add a collider to the rectangle
        rectangle.AddComponent<BoxCollider2D>();

        // Reset the mouse positions
        mouseDownPosition = Vector3.zero;
        mouseUpPosition = Vector3.zero;
    }

}

