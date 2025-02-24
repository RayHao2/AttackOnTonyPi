import tensorflow as tf
import numpy as np
import cv2
import os
import json
import shutil
from pycocotools.coco import COCO
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import decode
from core.utils import image_preprocess
from tensorflow.python.saved_model import tag_constants

# Command-line arguments
flags.DEFINE_string('model', './yolov4-tf', 'Path to YOLOv4 TensorFlow saved model')
flags.DEFINE_string('coco_dir', './coco', 'Path to COCO dataset')
flags.DEFINE_string('adv_output', './adv_image.jpg', 'Output path for adversarial image')
flags.DEFINE_float('epsilon', 0.03, 'Perturbation magnitude (ε)')
flags.DEFINE_float('alpha', 0.01, 'Step size for PGD')
flags.DEFINE_integer('iterations', 40, 'Number of PGD iterations')

# COCO class index for "person"
PERSON_CLASS_ID = 0  # COCO dataset labels person as class 0


#COCO classes 
# COCO Class Index to Name Mapping
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "TV", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}
YOLOV4_MODEL_PATH = os.path.abspath("./tensorflow-yolov4-tflite/yolov4-tf")

def load_yolo_model(model_path):
    """ Load the YOLOv4 model """
    if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
        raise FileNotFoundError(f"[ERROR] YOLOv4 model not found at {model_path}")

    print(f"[INFO] Loading YOLOv4 model from: {model_path}")
    model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    infer = model.signatures["serving_default"]
    return infer

def load_coco_image(coco_dir):
    """ 
    Load a random image from COCO dataset with non-human labels, 
    delete previous images in `image_dir`, and save the new image.
    """
    image_dir = "./image"
    # Step 1: Delete all existing images in `image_dir`
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)  # Remove the directory and all its contents
    os.makedirs(image_dir, exist_ok=True)  # Recreate the directory

    # Step 2: Load COCO dataset and filter out images containing people
    coco = COCO(os.path.join(coco_dir, "annotations/instances_val2017.json"))
    
    # Select only non-human images
    non_person_ids = []
    for cat in coco.loadCats(coco.getCatIds()):
        if cat['name'] != 'person':
            non_person_ids.extend(coco.getImgIds(catIds=[cat['id']]))

    # Step 3: Select a random image that does not contain a person
    image_info = coco.loadImgs(np.random.choice(non_person_ids, 1))[0]
    image_path = os.path.join(coco_dir, "val2017", image_info['file_name'])

    # Step 4: Load, resize, and normalize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (416, 416))
    image_normalized = image / 255.0  # Normalize
    image_normalized = np.expand_dims(image_normalized, axis=0).astype(np.float32)

    # Step 5: Save the image in `image_dir`
    saved_image_path = os.path.join(image_dir, image_info['file_name'])
    cv2.imwrite(saved_image_path, image)

    print(f"[INFO] Saved image to {saved_image_path}")

    return image_normalized, saved_image_path

def targeted_pgd_attack(model, image, target_class=0, epsilon=0.03, alpha=0.01, iterations=40):
    """ Perform targeted PGD attack on YOLOv4 model """
    adv_image = tf.Variable(image)

    for i in range(iterations):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(adv_image)
            preds = model(adv_image)

            if "tf.concat_16" in preds:  # Update to correct key
                logits = preds["tf.concat_16"]
            else:
                raise ValueError(f"[ERROR] Model output keys do not contain expected keys. Available keys: {preds.keys()}")

            # Targeted attack: Maximize the model's confidence in "person"
            loss = -tf.reduce_mean(logits[:, target_class])

        # Compute the gradient
        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)

        # PGD update step
        adv_image.assign_add(alpha * signed_grad)

        # Clip perturbations to be within the epsilon-ball
        adv_image.assign(tf.clip_by_value(adv_image, image - epsilon, image + epsilon))

    return adv_image.numpy()


def run_inference(model, image):
    """ Runs YOLOv4 on an image and extracts class predictions """
    preds = model(tf.constant(image))

    # Print all available keys in model output
    # print("[DEBUG] Model output keys:", preds.keys())

    # # Check for detection outputs
    # for key in preds.keys():
    #     print(f"[DEBUG] {key} -> Shape: {preds[key].shape}")

    # Find the correct key
    output_key = "tf.concat_16"  # Update with correct key from debug output

    if output_key in preds:
        logits = preds[output_key].numpy()
        print(f"[INFO] Found valid output: {output_key}")

        # Extract detected classes, confidence scores, and bounding boxes
        detected_classes, confidence_scores, bounding_boxes = extract_predictions(preds)

        # Convert class indices to names
        detected_class_names = [COCO_CLASSES.get(cls, "Unknown") for cls in detected_classes]

        print("[INFO] Final Detections:")
        for i, cls_name in enumerate(detected_class_names):
            print(f" - Object {i+1}: Class {cls_name} | Confidence: {confidence_scores[i]:.2f} | BBox: {bounding_boxes[i]}")

        return detected_class_names, confidence_scores, bounding_boxes
    else:
        raise ValueError(f"[ERROR] Model output keys do not match expected names. Available keys: {preds.keys()}")

def extract_predictions(preds):
    """ Extracts detected objects and class predictions from YOLO output. """
    output_key = "tf.concat_16"
    
    if output_key in preds:
        detections = preds[output_key].numpy()[0]  # Shape: (61, 84)
        bounding_boxes = detections[:, :4]  # First 4 values are bbox coordinates
        object_confidence = detections[:, 4]  # 5th value is object confidence
        class_probabilities = detections[:, 5:]  # Remaining values are class scores

        # Get the most confident class for each detected object
        detected_classes = np.argmax(class_probabilities, axis=1)
        confidence_scores = np.max(class_probabilities, axis=1)

        print("[INFO] Detected Classes:", detected_classes)
        print("[INFO] Confidence Scores:", confidence_scores)

        return detected_classes, confidence_scores, bounding_boxes

    else:
        raise ValueError(f"[ERROR] Model output keys do not contain expected tensor. Available keys: {preds.keys()}")

def main(_argv):
    tf.config.run_functions_eagerly(True)

    # Load YOLOv4 model
    infer = load_yolo_model(FLAGS.model)

    # Load a COCO image that is not a person
    image, image_name = load_coco_image(FLAGS.coco_dir)

    # Run YOLO on the original image and extract predictions
    print("[INFO] Running YOLOv4 on original image...")
    detected_classes, confidence_scores, bounding_boxes = run_inference(infer, image)





if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
