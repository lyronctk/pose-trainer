from matplotlib import pyplot as plt
from pathlib import Path
import mediapipe as mp
import numpy as np
from os import path
import argparse
import cv2

# BlazePose-specific utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
PL = mp_pose.PoseLandmark

# arguments
parser = argparse.ArgumentParser(description='Pose estimation for technique correction.')
parser.add_argument("--image-path", const=None, help='Path to target image.', required=True)
parser.add_argument("--annotation-save-path", const=None, help='Where to save annotated image. Omit if no annotation visuals needed.')
parser.add_argument("--vid-id", const=None, help='Video ID')
parser.add_argument("--frame-id", const=None, help='Frame ID')
parser.add_argument("--out-csv", const=None, help='CSV to append results to. Must have proper column format')

def main():
    print('hello world')

if __name__ == "__main__":
    args = parser.parse_args()
    im_path = args.image_path
    annotation_save_path = args.annotation_save_path
    vid_id = args.vid_id
    frame_id = args.frame_id
    out_csv = args.out_csv

    print(im_path)
    print(annotation_save_path) 
    print(vid_id) 
    print(frame_id) 
    print(out_csv) 
