export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hzg/TensorRT-5.1.5.0/lib
CUDA_VISIBLE_DEVICES=1 python demo_yolo3_deepsort.py ./video/testVideo.mp4 \
--yolo_cfg /home/hzg/code/deep_sort_pytorch/YOLOv3/cfg/yolo_v3.cfg \
--yolo_names YOLOv3/cfg/voc.names \
--yolo_weights /home/hzg/code/deep_sort_pytorch/YOLOv3/mymodel/yolov3.weights \
--conf_thresh 0.5 \
--ignore_display \
--deepsort_checkpoint /home/hzg/code/deep_sort_pytorch/deep_sort/deep_osnet/osnet_1.0.trt
