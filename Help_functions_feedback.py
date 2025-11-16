import numpy as np
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker
import subprocess
import os
from typing import List
import shutil
import cv2
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

