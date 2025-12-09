import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

def generate_bounding_box(save_dir, img_path, indices):
    # pre determined grid
    t_h, t_w = 10, 15 

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    for idx in indices:
        token_idx = int(idx) 
        row = token_idx // t_w
        col = token_idx %  t_w

        cell_h = img_h / t_h
        cell_w = img_w / t_w

        y0 = int(row * cell_h)
        y1 = int((row + 1) * cell_h)

        x0 = int(col * cell_w)
        x1 = int((col + 1) * cell_w)

        cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 1) 

        # add text label
        label = f"{token_idx}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4

        thickness = 1

        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size

        text_x = x0 + (x1 - x0 - text_w) // 2
        text_y = y1 - 4 # small offset

        text_y = min(text_y, img_h - 5)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    indices_str = "_".join(indices)
    save_path = os.path.join(save_dir, f"bboxes_{indices_str}.png")
    cv2.imwrite(save_path, img)
    
    return    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--indices", type=str, default=None) # "1,2,40" etc

    args = parser.parse_args()
    indices = args.indices.split(",")

    generate_bounding_box(args.save_dir, args.img_path, indices)


