Attadck on TonyPi


Deploy YOLOv4

1. download YOLOv4 weights and configuration
    - wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights
2. Convert Darknet Weights to TensorFlow
    - git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
3. Put the weights into the folder
    - mv yolov4.weights ensorflow-yolov4-tflite
4. cd tensorflow-yolov4-tflite
5. Convert the YOLOv4 model
    - python save_model.py --weights yolov4.weights --output ./yolov4-tf --input_size 416 --model yolov4



Download COCO dataset
1. make dir name coco
    -mkdir coco && cd coco
2. Download the zip file
    - wget http://images.cocodataset.org/zips/val2017.zip
    - wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
3. unzip val2017.zip
    - unzip val2017.zip
    - unzip annotations_trainval2017.zip


Notce: 
    - weights file and coco folder must must be in the tensorflow-yolov4-tflite folder

    - move attack.py in to the tensorflow-yolov4-tflite folder to work