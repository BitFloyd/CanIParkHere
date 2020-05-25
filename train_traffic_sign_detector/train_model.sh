# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=/Users/sebyjacob/My_Projects/CanIParkHere/train_traffic_sign_detector/tensorflow_object_detection/models/ssd_mobilenet_v2_coco.config
MODEL_DIR=/Users/sebyjacob/My_Projects/CanIParkHere/train_traffic_sign_detector/tensorflow_object_detection/models/ssd_mobilenet_v2_coco/trained_model
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
