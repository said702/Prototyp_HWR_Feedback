import cv2
from PIL import Image, ImageDraw, ImageFont
from math import atan2, degrees
import numpy as np


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

def draw_legend(frame):
    # Help text list (top → bottom)
    legend = [
    "h - Show/Hide Help",
    "ENTER - Start Feedback Process",
    "s - Switch to Spelling Feedback",
    "t - Switch to Translation Feedback",
    "o - Switch to Original Words",
    "x - Increase Font Size",
    "y - Decrease Font Size",
    "n - Reset Program",
    "q - Quit Program"
    ]


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    color = (0, 255, 255)  # Yellow

    x_offset = 20        # left margin
    y_start = 120        # <--- moved further down to avoid channel text
    line_height = 22     # distance between each help line

    # Draw each help line at the top-left, positioned lower
    for i, text in enumerate(legend):
        y = y_start + i * line_height
        cv2.putText(
            frame,
            text,
            (x_offset, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def draw_process_text(frame,text,font,font_scale,thickness):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  
    text_y = (frame.shape[0] + text_size[1]) // 2  
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 100, 0), thickness, cv2.LINE_AA)


        
