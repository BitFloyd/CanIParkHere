PIPELINE_CONFIG_PATH=/Users/sebyjacob/My_Projects/CanIParkHere/train_traffic_sign_detector/tensorflow_object_detection/models/ssd_mobilenet_v2_coco.config
MODEL_CHECKPOINT_PREFIX=/Users/sebyjacob/My_Projects/CanIParkHere/train_traffic_sign_detector/tensorflow_object_detection/models/ssd_mobilenet_v2_coco_tr/trained_model/model.ckpt-50000
SAVE_EXPORTED_PATH=/Users/sebyjacob/My_Projects/CanIParkHere/train_traffic_sign_detector/tensorflow_object_detection/models/ssd_mobilenet_v2_coco_tr/trained_model/exported_graph

python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${MODEL_CHECKPOINT_PREFIX} \
    --output_directory ${SAVE_EXPORTED_PATH}

