import cv2
import os

def vid2frames(videopath, outputpath):
    vidcap = cv2.VideoCapture(videopath)
    success,image = vidcap.read()
    count = 0
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    while success:
      cv2.imwrite(os.path.join(outputpath, "frame%d.jpg"% count), image)
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1

def main():
    rootdir = '/home/amolsingh/gcloud/data/videos/'
    outdir = '/home/amolsingh/gcloud/data/frames/'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            error = os.path.split(subdir)[1]
            if file.endswith('.MOV') or file.endswith('.mov'):
                vid2frames(os.path.join(subdir, file), os.path.join(outdir, error, file[:len(file)-4]))
                #print(os.path.join(subdir, file))
                print(os.path.join(outdir, error, file[:len(file)-4]))
        
main()