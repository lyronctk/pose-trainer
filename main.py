import argparse
import os

model_names = ['VP3D', 'BP']
output_path = ''
def parse_args():
    """
    Get input data directory and pose estimation model choice
    """
    parser = argparse.ArgumentParser(description='Trainer Inference')
    
    parser.add_argument('inputdata', metavar='IDIR', help='path to input data') 
    parser.add_argument('-outputdir', metavar='ODIR', help='path for output data of detectron') 
    parser.add_argument('-m', '--model', metavar='ARCH', default='VP3D', choices=model_names, help='model selection: ' + ' | '.join(model_names) + ' (default: VP3D)')
    args = parser.parse_args()
    return args


def run_PE(inputdata, outputdir, model):
    """
    run the pose estimation
    """
    if model == 'VP3D':
        os.system("cd pose_estimation/video_pose_3d/inference")
        os.system("python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir {} \
    --image-ext mp4 \
    {}".format(outputdir, inputdata))
        os.system("cd pose_estimation/video_pose_3d/data")
        os.system("python prepare_data_2d_custom.py -i {} -o myvideos".format(outputdir))
        os.system("python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject input_video.mp4 --viz-action custom --viz-camera 0 --viz-video /path/to/input_video.mp4 --viz-output output.mp4 --viz-size 6".format(outputdir))
def main(args):
    model = run_PE(args.inputdata, args.m)

if __name__ == "__main__":
    args = parse_args()
    main(args)
