import os
import cv2

PROJ_ROOT= '/home/amolsingh/gcloud/trainer/'
outcsv = PROJ_ROOT+'data/class2.csv'
rootdir = PROJ_ROOT+'data/frames/2_squat'

for subdir, dirs, files in os.walk(rootdir):
    vid_id = os.path.split(subdir)[1]
    for file in files: 
        img_path = os.path.join(subdir, file)
        os.system("python /home/amolsingh/gcloud/trainer/form_classification/clf_rule_set.py --image-path {} --vid-id {} --frame-id {} --out-csv {}".format(img_path, vid_id, file, outcsv))