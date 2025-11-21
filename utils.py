import numpy as np
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker
import subprocess
import os
from typing import List
import shutil
import cv2
from PIL import Image, ImageDraw, ImageFont
from math import atan2, degrees


spell = SpellChecker()



def save_bounding_boxes(boxes, orig_image):
    output_path = "extracted_bboxes_for_prediction"
    if os.path.exists(output_path):
      shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    image = cv2.imread(orig_image)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        offset=1
        cropped_image = image[(y1-offset):(y2+offset), (x1-offset):(x2+offset)]
        try:
          resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
          resized_image = cropped_image  # Fallback auf cropped_image

        output_file = os.path.join(output_path, f"bbox_{idx + 1}.jpg")
        try:
          cv2.imwrite(output_file, resized_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        except:
          pass 



def extract_bounding_boxes(boxes) -> List[List[int]]:
    if boxes.xyxy.numel() == 0:
        return []
    bboxes = boxes.xyxy.cpu().numpy().tolist()
    formatted_bboxes = [[int(x1), int(y1), int(x2), int(y2), int(x1), int(y2), int(x2), int(y1)] for x1, y1, x2, y2 in bboxes]
    return formatted_bboxes




def safe_translate(word):
    try:
        translated_word = GoogleTranslator(source='auto', target='de').translate(word)
        return translated_word
    except Exception as e:
        return word

def multi_frame_denoising(curr_frame, frame_queue):
    frame_queue.append(curr_frame)
    denoised_frame = np.mean(frame_queue, axis=0).astype(np.uint8)
    return denoised_frame


def feedback_word_speeling_and_translator(orginal_words):
    speeling_words_list = []
    translation_words=[]
    for i,word in enumerate(orginal_words):
      word_corrected=spell.correction(word)
      if(word_corrected==None):
         word_corrected=word
      speeling_words_list.append(word_corrected)
      if speeling_words_list[i] in [',','.', '?', '!']:
        translation_words.append(speeling_words_list[i]) 
      else:
        t_w=safe_translate(speeling_words_list[i])
        translation_words.append(t_w)
    return speeling_words_list,translation_words


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0




def update_bbs(prev_bbs, curr_bbs, iou_threshold=0.5, diff_threshold=5, global_iou_threshold=0.1):
    updated_bbs = []
    used_indices = set()
    iou_values = []
    for prev_box in prev_bbs:
        matched = False
        for idx, curr_box in enumerate(curr_bbs):
            if idx in used_indices:
                continue
            iou = calculate_iou(prev_box, curr_box)
            iou_values.append(iou)

            if iou >= iou_threshold:
                diff = np.mean([abs(p - c) for p, c in zip(prev_box[:4], curr_box[:4])])
                if diff > diff_threshold:
                    updated_bbs.append(curr_box)
                else:
                    updated_bbs.append(prev_box)
                used_indices.add(idx)
                matched = True
                break
        if not matched:
            updated_bbs.append(prev_box)
    avg_iou = np.mean(iou_values) if iou_values else 0.0
    if avg_iou < global_iou_threshold:
        return curr_bbs 
    return updated_bbs



def calculate_filtered_ious(curr_bbs, x_tolerance=50, y_tolerance=50, iou_threshold=0.05, remove=True):
    iou_results = []
    def are_bbs_near(bb1, bb2):
        return (
            abs(bb1[0] - bb2[0]) <= x_tolerance and 
            abs(bb1[1] - bb2[1]) <= y_tolerance     
        )

    for i, box1 in enumerate(curr_bbs):
        for j, box2 in enumerate(curr_bbs):
            if i < j:  
                if are_bbs_near(box1, box2):  
                    iou = calculate_iou(box1, box2)
                    if iou >= iou_threshold: 
                        iou_results.append((i, j, iou))
    if remove:
        to_remove = set()
        for i, j, iou in iou_results:
            box1 = curr_bbs[i]
            box2 = curr_bbs[j]

            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            if area1 > area2:
                to_remove.add(i)
            else:
                to_remove.add(j)
        curr_bbs = [box for idx, box in enumerate(curr_bbs) if idx not in to_remove]

    return curr_bbs if remove else iou_results


def draw_text_comparison(image,font_scale2 ,original_words, feedback_words, bboxes, color ,erweitere,speeling_feedback=0,translation=0,thickness=1):
    color_same =  color
    color_diff = (0, 0, 255)  # Rot für unterschiedliche Buchstaben

    for i, box in enumerate(bboxes):
        if i >= len(feedback_words):
            break

        original_word = original_words[i] if i < len(original_words) else ""
        feedback_word = feedback_words[i]
        x1, y1, x2, y2, x3, y3, x4, y4 = box
      
        if feedback_word is None:
          feedback_word = original_word
          
        #Um BBs anzuzeigen 
        #cv2.polylines(image, [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Mittlere X-Position 
        bbox_center_x = (x1 + x2) // 2

        #Höhe der Bounding Box, um die Schriftgröße dynamisch zu skalieren
        bbox_height = abs(y1 - y3)  # Vertikaler Unterschied
        bbox_width = abs(x2 - x1)   # Horizontaler Unterschied

        font_scale = font_scale2
  
        min_length = min(len(original_word), len(feedback_word))
     
        text_size = cv2.getTextSize(feedback_word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        position_x = bbox_center_x - text_size[0] // 2  
        vertical_offset = text_size[1]  
        position_y = max(y3, y4) + vertical_offset
       
        if(len(feedback_word)>len(original_word) and erweitere==1 ):
          diff_length = len(feedback_word) - len(original_word)
          original_word = diff_length*' '+original_word

        if speeling_feedback == 1:
            for j in range(len(feedback_word)):
                if j < min_length and original_word[j] == feedback_word[j]:
                    cv2.putText(image, feedback_word[j], (position_x, position_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_same, thickness, cv2.LINE_AA)
                else:
                    cv2.putText(image, feedback_word[j], (position_x, position_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_diff, thickness, cv2.LINE_AA)

                char_size = cv2.getTextSize(feedback_word[j], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                position_x += char_size[0]

        else:
            cv2.putText(image, feedback_word, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_same, thickness, cv2.LINE_AA)


def process_frame(frame, bbs,ocr_running,Feedback_running,Feedback_done):
    frame_copy = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not bbs:
        # Keine Bounding Boxes gefunden
        text = "No handwriting detected"
        font_scale = 2
        font_color = (0, 0, 255)  # Rot
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame_copy, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    else:
        # Bounding Boxes vorhanden
        for bb in bbs:
            x1, y1, x2, y2 = bb[:4]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = "Handwriting detected"
        font_scale = 2
        font_color = (85, 107, 47)  # Grün
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2  
        text_y = (frame.shape[0] + text_size[1]) // 2  
        cv2.putText(frame_copy, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    
    return frame_copy



def draw_legend(frame, font_scale=0.8, thickness=2):
    """
    Draw the interactive legend at the bottom-left corner of the frame.
    Items are listed from bottom to top.
    """
    legend = [
        "q - Quit Program",
        "n - Reset Program",
        "y - Decrease Font Size",
        "x - Increase Font Size",
        "o - Switch to Original Words",
        "t - Switch to Translation Feedback",
        "s - Switch to Spelling Feedback",
        "ENTER - Start Feedback Process",
        "h - Show/Hide Help"
        ]
    x_offset = 10  # Left margin

    # Dynamically compute text height
    (text_w, text_h), _ = cv2.getTextSize(
        "Sample", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    line_height = text_h + 10  # Add small spacing

    # Start from bottom-left and go upwards
    y_start = frame.shape[0] - 20  # margin from bottom

    for i, text in enumerate(legend):
        y = y_start - i * line_height
        cv2.putText(
            frame,
            text,
            (x_offset, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 140, 255),  # Dark orange (BGR)
            thickness,
            cv2.LINE_AA,
        )


def draw_process_text(frame,text,font,font_scale,thickness):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  
    text_y = (frame.shape[0] + text_size[1]) // 2  
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 100, 0), thickness, cv2.LINE_AA)


        
