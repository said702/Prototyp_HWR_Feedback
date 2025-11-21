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



## Starting the Prototype

Start the application using:
  ```bash
   poetry run python main.py --camera <index>
   ```
  
### Which camera index should you use?

- If your laptop has an internal webcam AND you plug in a USB/document camera:
    → Use --camera 1

- If you only use one external camera (no internal webcam active):
    → Use --camera 0

------------------------------------------------------------

### Keyboard Controls

While the prototype is running and the camera window is active, you can control the system with these keys:

| Key      | Action |
|----------|--------|
| **ENTER** | Start the handwriting recognition and feedback process |
| **s**     | Switch to *Spelling Feedback* mode (highlight character differences) |
| **t**     | Switch to *Translation Feedback* mode (translate words from English to German) |
| **o**     | Show *Original Words* (display the recognized words without feedback) |
| **x**     | Increase feedback font size |
| **y**     | Decrease feedback font size |
| **n**     | Reset the prototype (clear internal state) |
| **q**     | Quit the program |
| **h**     | Show/Hide the on-screen help legend in the camera image |

⚠️ **Note:** If the keyboard shortcuts do not work, click once inside the camera window to give it focus.

