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

# Own modules (assumed to exist and work as given)
from Feedback_mapper import (
    draw_process_text,
    draw_text_comparison,
    process_frame,
    draw_legend,
)
from Help_Data_Converter import createDataset, generate_gt_file
from Help_functions_feedback import (
    extract_bounding_boxes,
    save_bounding_boxes,
    safe_translate,
    feedback_word_speeling_and_translator,
    multi_frame_denoising,
    update_bbs,
    calculate_iou,
    calculate_filtered_ious,
)
from model.dataset import hierarchical_dataset, AlignCollate
from model.Modell_S import initialize_model, predict


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Camera ID (change this if you use a different camera)
camera_id = 1

# YOLO model path
yolo_model_path = "Detection_Model/best.pt"

# AttentionHTR model path
attention_htr_model_path = "Extraction_Model/AttentionHTR.pth"

# Base directory for temporary results (images, texts, bounding boxes)
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

# Create results directory for temporary images, texts, and bounding boxes
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

image_path = os.path.join(base_dir, f"Image_{now_str}.jpg")


# -------------------------------------------------------------------------
# Global state for OCR and feedback
# -------------------------------------------------------------------------

ocr_running = False
Feedback_running = False
ocr_done = False
Feedback_done = False

orginal_words = []  # (kept name as in your original code)
feedback_words = []
translated_words_list = []
speeling_words_list = []
speeling_feedback = 1
bboxes = []
bbs_paragraph = []
translated_feedback_text_list = []
translation_words = []
Kanal = ""
bbs_show = []
translation = 0
is_active = 0
rotate = 0
Paragraphs = []

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 2
color = (255, 0, 0)
font_scale2 = 0.8
erweitere = 0
frame_count = 0
Hilfe = True  # Show help legend

# Tracking / smoothing
prev_frame = None
prev_bbs_yolo11 = None
frame_queue = deque(maxlen=5)
diff_threshold = 5
prev_BBs_for_Prediction = 0
all_image_crops_and_bboxes = []

# Some additional variables that are reset but not otherwise used
arrangement_show = False
predicted_classes = []

# Last original frame (used by HWR thread)
frame_orig = None


# -------------------------------------------------------------------------
# Function to run HWR and feedback generation in a background thread
# -------------------------------------------------------------------------

def run_hwr_and_feedback(bbs_yolo11):
    """
    Runs the handwriting recognition (HWR) and feedback generation pipeline
    in a background thread.
    """
    global ocr_running
    global ocr_done
    global Feedback_running
    global Feedback_done
    global orginal_words
    global speeling_words_list
    global font_scale2
    global translated_words_list
    global feedback_words
    global bboxes
    global bbs_paragraph
    global Kanal
    global translated_feedback_text_list
    global bbs_show
    global translation_words
    global Paragraphs
    

    ocr_running = True

    # Save current frame and bounding boxes for processing
    cv2.imwrite(image_path, frame_orig)
    save_bounding_boxes(bbs_yolo11, image_path)

    # Generate GT file and LMDB dataset for AttentionHTR
    gt_file = generate_gt_file()
    createDataset(gt_file)

    # Build evaluation dataset and dataloader
    eval_data, eval_data_log = hierarchical_dataset(root="lmdb_dataset", opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=AlignCollate_evaluation,
        pin_memory=True,
    )

    # Predict with AttentionHTR
    predictions = predict(model, evaluation_loader, converter, opt)

    orginal_words = []
    for pred in predictions:
        orginal_words.append(pred)

    bboxes = bbs_yolo11

    # Switch state from OCR to feedback generation
    ocr_running = False
    ocr_done = True
    Feedback_running = True
    Feedback_done = False

    # Create spelling feedback and translations at word level
    speeling_words_list, translation_words_local = feedback_word_speeling_and_translator(
        orginal_words
    )

    Feedback_running = False
    Feedback_done = True

    feedback_words = speeling_words_list
    Kanal = "Spelling Feedback"
    bbs_show = bboxes
    # Update global translation words
    translation_words[:] = translation_words_local


# -------------------------------------------------------------------------
# Open camera
# -------------------------------------------------------------------------

cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Prototype", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Prototype", 1280, 720)


if not cap.isOpened():
    print(f"Error: Could not open camera with ID {camera_id}.")
    raise SystemExit(1)


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    # Keep original frame for HWR processing
    frame_orig = frame.copy()

    # Multi-frame denoising
    frame = multi_frame_denoising(frame, frame_queue)
    frame_den = frame  # If you need it later; kept for compatibility

    frame_count += 1

    # Run detection every 3rd frame or on the first frame when nothing is running
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
        else:
            bbs_yolo11 = bbs_yolo11

        # If new bounding boxes appear (more than before), reset states
        if frame_count > 1 and prev_bbs_yolo11 is not None:
            if len(bbs_yolo11) > len(prev_bbs_yolo11):
                ocr_running = False
                Feedback_running = False
                ocr_done = False
                Feedback_done = False
                orginal_words = []
                feedback_words = []
                translated_words_list = []
                speeling_words_list = []
                speeling_feedback = 1
                bboxes = []
                bbs_paragraph = []
                translated_feedback_text_list = []
                translation_words = []
                Kanal = ""
                bbs_show = []
                translation = 0
                is_active = 0
                rotate = 0
                Paragraphs = []
                erweitere = 0
                arrangement_show = False
                all_image_crops_and_bboxes = []
                predicted_classes = []

        prev_bbs_yolo11 = bbs_yolo11

    # Filter bounding boxes by IoU and tolerance
    bbs_yolo11 = calculate_filtered_ious(
        bbs_yolo11, x_tolerance=200, y_tolerance=200, iou_threshold=0.1, remove=True
    )

    # ---------------------------------------------------------------------
    # Drawing based on current state
    # ---------------------------------------------------------------------

    if ocr_running:
        text = "Handwriting recognition is running..."
        draw_process_text(frame, text, font, font_scale, thickness)

    if Feedback_running and ocr_done:
        # Show original predictions while feedback is being generated
        draw_text_comparison(
            frame,
            font_scale2,
            orginal_words,
            orginal_words,
            bboxes,
            color,
            0,
            thickness=2,
            translation=0,
        )
        text = "Feedback is being generated..."
        draw_process_text(frame, text, font, font_scale, thickness)

    if ocr_done and Feedback_done:
        # Show final feedback (spelling or translation, depending on current mode)
        draw_text_comparison(
            frame,
            font_scale2,
            orginal_words,
            feedback_words,
            bbs_show,
            color,
            erweitere,
            speeling_feedback,
            translation,
            thickness=2,
        )
        cv2.putText(
            frame,
            Kanal,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # When nothing is running or finished, show interactive bounding boxes, etc.
    if (ocr_running + Feedback_running + Feedback_done) == 0:
        frame = process_frame(frame, bbs_yolo11, ocr_running, Feedback_running, Feedback_done)

    # Draw legend / help if enabled
    if Hilfe:
        draw_legend(frame)

    # Show window
    cv2.imshow("Prototype", frame)
    prev_frame = frame.copy()

    # ---------------------------------------------------------------------
    # Key handling
    # ---------------------------------------------------------------------
    key = cv2.waitKey(1)

    # ENTER key: start HWR and feedback in a background thread
    if key in [ord("\r"), ord("\n")] and not ocr_running:
        thread = threading.Thread(target=run_hwr_and_feedback, args=(bbs_yolo11,))
        thread.start()

    # 's' key: show spelling feedback
    elif key == ord("s") and ocr_done and Feedback_done:
        feedback_words = speeling_words_list
        speeling_feedback = 1
        translation = 0
        Kanal = "Spelling Feedback"
        color = (255, 0, 0)
        bbs_show = bboxes
        rotate = 0

    # 'h' key: toggle help legend
    elif key == ord("h"):
        Hilfe = not Hilfe

    # 'e' key: toggle extended text mode (erweitere)
    elif key == ord("e"):
        erweitere = not erweitere

    # 'x' key: increase feedback font scale
    elif key == ord("x"):
        font_scale2 += 0.125

    # 'y' key: decrease feedback font scale
    elif key == ord("y"):
        font_scale2 -= 0.125

    # 't' key: show translation feedback
    elif key == ord("t") and ocr_done and Feedback_done:
        feedback_words = translation_words
        bbs_show = bboxes
        Kanal = "Translation Feedback"
        speeling_feedback = 0
        translation = 1
        color = (255, 0, 0)

    # 'o' key: show original predictions
    elif key == ord("o") and ocr_done and Feedback_done:
        feedback_words = orginal_words
        speeling_feedback = 0
        translation = 0
        Kanal = "Original"
        color = (255, 0, 0)
        bbs_show = bboxes

    # 'n' key: reset everything
    elif key == ord("n"):
        ocr_running = False
        Feedback_running = False
        ocr_done = False
        Feedback_done = False
        orginal_words = []
        feedback_words = []
        translated_words_list = []
        speeling_words_list = []
        speeling_feedback = 1
        bboxes = []
        bbs_paragraph = []
        translated_feedback_text_list = []
        translation_words = []
        Kanal = ""
        bbs_show = []
        translation = 0
        is_active = 0
        rotate = 0
        Paragraphs = []
        erweitere = 0
        arrangement_show = False
        all_image_crops_and_bboxes = []
        predicted_classes = []

    # 'q' key: quit
    elif key == ord("q"):
        break

# -------------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
