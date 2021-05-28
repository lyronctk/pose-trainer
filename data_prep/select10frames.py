import cv2
import os
import shutil
import random
import pandas as pd

def select10frames(mdata, splitGroup, aVidFramesPath, outputpath, errorClass, videoID):
    filenames = random.sample(os.listdir(aVidFramesPath), 10)
    for count, fname in enumerate(filenames):
        srcpath = os.path.join(aVidFramesPath, fname)
        #print("src: ", srcpath)
        img = cv2.imread(srcpath)
        if not os.path.exists(os.path.join(outputpath, errorClass, videoID)):
            os.makedirs(os.path.join(outputpath, errorClass, videoID))
        outpth = os.path.join(outputpath, errorClass, videoID, fname)
        cv2.imwrite(outpth, img)
        print("out:", outpth)
        pdT = pd.DataFrame([[outpth, splitGroup, errorClass]], columns=['Path', 'Split Group', 'Class Label'])        
        mdata = mdata.append(pdT, ignore_index=True)
    return mdata
        
        
        
        #shutil.copy(srcpath, os.path.join(outputpath, errorClass, videoID))
        
def main():
    rootdir = '/home/amolsingh/gcloud/trainer/data/frames/'
    outdir = '/home/amolsingh/gcloud/trainer/data/10frames/'
    csv = '/home/amolsingh/gcloud/trainer/data/metadata.csv'
    #mdata = pd.read_csv(csv)
    count = 0
    mdata = pd.DataFrame(columns=['Path', 'Split Group', 'Class Label'])
    for subdir, dirs, files in os.walk(rootdir):
        vidID = os.path.split(subdir)[1]
        error = os.path.split(os.path.split(subdir)[0])[1]
        if len(files) > 0:
            count += 1
            if count%18 <= 11:
                mdata = select10frames(mdata, "train", subdir, outdir, error, vidID)
            elif count%18 > 11 and count%18 < 14:
                mdata = select10frames(mdata, "val", subdir, outdir, error, vidID)
            elif count%18 >= 14:
                mdata = select10frames(mdata, "test", subdir, outdir, error, vidID)
    print(mdata)
    mdata.to_csv(csv, index=False)
                
main()