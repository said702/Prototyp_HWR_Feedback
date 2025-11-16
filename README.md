### Handwriting Feedback Prototype

This prototype captures a live video stream from a camera, detects handwritten words, recognizes them, and overlays feedback (spelling and translation) directly on the image.

The system combines:

- **Text Detection:** Ultralytics YOLO (YOLO11 variant)  
  - Official repository: https://github.com/ultralytics/ultralytics  
  - The YOLO detection model used in this prototype was trained by us on the IAM and Imgur5K datasets.
- **Text Recognition:** Attention-based HTR model (AttentionHTR)  
  - Original repository: https://github.com/dmitrijsk/AttentionHTR  
  - The AttentionHTR model weights are downloaded automatically using Google Drive:  
    https://drive.google.com/drive/folders/1RBWJ9TkjI8hG0BIBjJNeZLRcEysFBYds?usp=sharing


### Installation & Dependencies

This project uses **Poetry** to manage dependencies and virtual environments.

1. Install Poetry (if not installed):

   ```bash
   pip install poetry
   ```

2. Install project dependencies (run this in the folder where `pyproject.toml` is located):

   ```bash
   poetry install
   ```

This will:

- create or reuse a virtual environment  
- install all required packages (OpenCV, Ultralytics, Torch, Pillow, gdown, etc.)  
- set everything up for running the prototype

### Starting the Prototype

To start the system, run in the project directory:

```bash
poetry run python Main.py
```

A window will appear showing the live camera feed.


### Camera Selection

If you have multiple cameras (webcam + document camera), you can select which one to use by setting the `camera_id` in `Main.py`:

```python
camera_id = 0
cap = cv2.VideoCapture(camera_id)
```

Typical IDs:

- `0` → built-in laptop camera  
- `1` → USB/document camera  
- `2`, `3`, `4`, ... → additional cameras  

If the wrong camera shows up or the screen is black, change the `camera_id` value, save the file, and restart with:

```bash
poetry run python Main.py
```


### Keyboard Controls

While the prototype is running and the camera window is active, you can control the system with these keys:

| Key      | Action |
|----------|--------|
| **ENTER** | Start the handwriting recognition and feedback process |
| **s**     | Switch to *Spelling Feedback* mode (highlight character differences) |
| **t**     | Switch to *Translation Feedback* mode (show translated words under the bounding boxes) |
| **o**     | Show *Original Words* (recognized words without feedback coloring) |
| **e**     | Extend original words with spaces if they are shorter than feedback words (alignment) |
| **x**     | Increase feedback font size |
| **y**     | Decrease feedback font size |
| **n**     | Reset the prototype (clear internal state) |
| **q**     | Quit the program |
| **h**     | Show/Hide the on-screen help legend in the camera image |
