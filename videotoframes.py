import cv2


def vid2frames(videopath, outputpath):
    vidcap = cv2.VideoCapture(videopath)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(outputpath+"frame%d.jpg" % count, image)
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1

vid2frames('/Users/amolsingh/Downloads/ins/lyron_deg45_take1.mp4','/Users/amolsingh/Downloads/outs')