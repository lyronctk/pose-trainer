# Project constants
PROJ_ROOT=/Users/lyronctk/Desktop/projects/trainer
REF_IM_DIR=${PROJ_ROOT}/data/rule_set_references

# Run params
IMAGE_PATH=${REF_IM_DIR}/ref_lockout.png
ANNOTATION_PATH=${REF_IM_DIR}/ref_lockout_annotated.png
VID_ID=1
FRAME_ID=1
OUT_CSV=${REF_IM_DIR}/predictions.csv

# CMD
python $PROJ_ROOT/form_classification/clf_rule_set.py --image-path $IMAGE_PATH \
                                                      --annotation-save-path $ANNOTATION_PATH \
                                                      --vid-id $VID_ID \
                                                      --frame-id $FRAME_ID \
                                                      --out-csv $OUT_CSV \
