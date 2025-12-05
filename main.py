import os
import cv2
import threading
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import gdown
import argparse
import ctypes 
import sys
from screeninfo import get_monitors



# -------------------------------------------------------------------------
# Argument parser for camera index
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Prototype Demo")
parser.add_argument(
    "--camera",
    type=int,
    default=1,
    help="Camera index to use (default: 1)"
)
args = parser.parse_args()
camera_index = args.camera


# Own modules
from utils import (
    draw_process_text,
    draw_text_comparison,
    process_frame,
    draw_legend,
    extract_bounding_boxes,
    save_bounding_boxes,
    feedback_word_speeling_and_translator,
    multi_frame_denoising,
    update_bbs,
    calculate_filtered_ious,
)

from help_data_converter import createDataset, generate_gt_file
from model.dataset import hierarchical_dataset, AlignCollate
from model.Modell_S import initialize_model, predict


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

yolo_model_path = "Detection_Model/best.pt"
attention_htr_model_path = "Extraction_Model/AttentionHTR.pth"
base_dir = "Results"


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

print("Loading YOLO11 detection model...")
trained_model = YOLO(yolo_model_path)
print("YOLO11 model loaded successfully.")

print("Loading AttentionHTR model...")
if not os.path.exists(attention_htr_model_path):
    print("AttentionHTR model not found locally. Downloading from Google Drive...")
    file_id = "1IpwLT0qi8cDB7kzBISFg3V7stHxm5UgW"
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(attention_htr_model_path), exist_ok=True)
    gdown.download(url, attention_htr_model_path, quiet=False)

model, converter, criterion, AlignCollate_evaluation, opt = initialize_model()
print("AttentionHTR model loaded successfully.")

if torch.cuda.is_available():
    print("GPU available")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")


# -------------------------------------------------------------------------
# Timestamp and directories
# -------------------------------------------------------------------------

now = datetime.now()
now_str = now.strftime("%Y_%m_%d_%H_%M_%S")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

image_path = os.path.join(base_dir, f"Image_{now_str}.jpg")


# -------------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------------

ocr_running = False
Feedback_running = False
ocr_done = False
Feedback_done = False

orginal_words = []
feedback_words = []
speeling_words_list = []
translated_words_list = []
translation_words = []
bboxes = []
bbs_paragraph = []

Kanal = ""
bbs_show = []
translation = 0
speeling_feedback = 1
erweitere = 0
is_active = 0
rotate = 0
Paragraphs = []

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_scale2 = 0.8
thickness = 2
color = (255, 0, 0)

frame_count = 0
Hilfe = True

prev_frame = None
prev_bbs_yolo11 = None
frame_queue = deque(maxlen=5)
diff_threshold = 5
all_image_crops_and_bboxes = []


# -------------------------------------------------------------------------
# Thread: HWR + Feedback
# -------------------------------------------------------------------------

def run_hwr_and_feedback(bbs_yolo11):
    global ocr_running, ocr_done, Feedback_running, Feedback_done
    global orginal_words, feedback_words, speeling_words_list
    global translated_words_list, translation_words
    global bboxes, bbs_show, Kanal

    ocr_running = True

    cv2.imwrite(image_path, frame_orig)
    save_bounding_boxes(bbs_yolo11, image_path)

    gt_file = generate_gt_file()
    createDataset(gt_file)

    eval_data, eval_data_log = hierarchical_dataset(root="lmdb_dataset", opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=AlignCollate_evaluation,
        pin_memory=True,
    )

    predictions = predict(model, evaluation_loader, converter, opt)
    orginal_words = [p for p in predictions]

    bboxes = bbs_yolo11

    ocr_running = False
    ocr_done = True
    Feedback_running = True

    speeling_words_list, translation_words_local = feedback_word_speeling_and_translator(
        orginal_words
    )

    Feedback_running = False
    Feedback_done = True

    feedback_words = speeling_words_list
    Kanal = "Spelling Feedback"
    bbs_show = bboxes
    translation_words[:] = translation_words_local


# -------------------------------------------------------------------------
# Screen + Camera
# -------------------------------------------------------------------------

m = get_monitors()[0]
screen_width  = m.width
screen_height = m.height

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"[ERROR] Could not open camera with index {camera_index}.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Prototyp", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Prototyp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.resizeWindow("Prototyp", screen_width, screen_height)
cv2.moveWindow("Prototyp", 0, 0)


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    frame_orig = frame.copy()

    frame = multi_frame_denoising(frame, frame_queue)
    frame_display = frame.copy()            

    frame_count += 1

    # ------------------ YOLO Detection ------------------
    if frame_count % 3 == 0 or (
        frame_count == 1 and (ocr_running + Feedback_running + Feedback_done) == 0
    ):

        Detection_Results = trained_model(frame, conf=0.35, verbose=False)
        bbs_yolo11 = extract_bounding_boxes(Detection_Results[0].boxes)

        if prev_bbs_yolo11 is not None:
            bbs_yolo11 = update_bbs(
                prev_bbs_yolo11,
                bbs_yolo11,
                iou_threshold=0.8,
                diff_threshold=diff_threshold,
                global_iou_threshold=0.1,
            )

        prev_bbs_yolo11 = bbs_yolo11

    bbs_yolo11 = calculate_filtered_ious(
        bbs_yolo11, x_tolerance=200, y_tolerance=200, iou_threshold=0.1, remove=True
    )


    # ------------------ DRAWING AREA ------------------

    if ocr_running:
        draw_process_text(frame_display, "Handwriting recognition is running...", font, font_scale, thickness)

    if Feedback_running and ocr_done:
        draw_text_comparison(
            frame_display, font_scale2, orginal_words, orginal_words,
            bboxes, color, 0, thickness=2, translation=0
        )
        draw_process_text(frame_display, "Feedback is being generated...", font, font_scale, thickness)

    if ocr_done and Feedback_done:
        draw_text_comparison(
            frame_display, font_scale2, orginal_words, feedback_words,
            bbs_show, color, erweitere, speeling_feedback, translation, thickness=2
        )
        cv2.putText(frame_display, Kanal, (10, 50), font, 1.5, (0, 0, 0), 2)


    if (ocr_running + Feedback_running + Feedback_done) == 0:
        frame_display = process_frame(frame_display, bbs_yolo11, ocr_running, Feedback_running, Feedback_done)


    if Hilfe:
        draw_legend(frame_display)


    # ------------------ SHOW FRAME ------------------
    cv2.imshow("Prototyp", frame_display)  


    # ------------------ KEYS ------------------
    key = cv2.waitKey(1)

    if key in [ord("\r"), ord("\n")] and not ocr_running:
        thread = threading.Thread(target=run_hwr_and_feedback, args=(bbs_yolo11,))
        thread.start()

    elif key == ord("s") and ocr_done and Feedback_done:
        feedback_words = speeling_words_list
        speeling_feedback = 1
        translation = 0
        Kanal = "Spelling Feedback"
        bbs_show = bboxes

    elif key == ord("t") and ocr_done and Feedback_done:
        feedback_words = translation_words
        speeling_feedback = 0
        translation = 1
        Kanal = "Translation Feedback"
        bbs_show = bboxes

    elif key == ord("o") and ocr_done and Feedback_done:
        feedback_words = orginal_words
        speeling_feedback = 0
        translation = 0
        Kanal = "Original"
        bbs_show = bboxes

    elif key == ord("h"):
        Hilfe = not Hilfe

    elif key == ord("e"):
        erweitere = not erweitere

    elif key == ord("x"):
        font_scale2 += 0.125

    elif key == ord("y"):
        font_scale2 -= 0.125

    elif key == ord("n"):
        # full reset
        ocr_running = False
        Feedback_running = False
        ocr_done = False
        Feedback_done = False
        orginal_words = []
        feedback_words = []
        speeling_words_list = []
        translation_words = []
        translated_words_list = []
        bboxes = []
        bbs_show = []
        Kanal = ""
        translation = 0
        erweitere = 0

    elif key == ord("q"):
        break


# -------------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
