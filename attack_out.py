import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
# from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import foolbox as fb
import eagerpy as ep
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


def load_yolo_model(model_path):
    """
    Load the YOLOv4 model from the saved TensorFlow checkpoint.
    """
    print(f"[INFO] Loading YOLOv4 model from: {model_path}")

    # Load the saved model
    model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])

    # Extract the inference function
    infer = model.signatures['serving_default']

    
    print("[INFO] Model loaded successfully!")

    # # Print model signature details
    # print("\n[INFO] Model Signature Keys:", model.signatures.keys())
    # print("\n[INFO] Model Inputs:")
    # for key, value in infer.structured_input_signature[1].items():
    #     print(f"  - {key}: {value}")

    # print("\n[INFO] Model Outputs:")
    # for key, value in infer.structured_outputs.items():
    #     print(f"  - {key}: {value}")
    
    return infer




def pgd_attack_yolo_tf(model, image, base_class_id, target_class_id, epsilon=0.03, alpha=0.01, iterations=40):
    """
    Step 1: Initialize random perturbation δ within [-ε, ε]
    """

def print_model_layers(model):
    print("[INFO] Printing model layer operations:")
    
    # Extract the inference function
    infer = model.signatures['serving_default']

    # List all operations in the computational graph
    for op in infer.graph.get_operations():
        print(op.name)  # Print layer names


# def foolbox_attack(model, image, epsilon=0.03):
#     """
#     Generate an adversarial image using Foolbox by extracting labels from YOLO predictions.
#     """



def extract_yolo_detections(model, image):
    """
    Runs YOLO inference and extracts class labels and confidence scores correctly.
    """
    # Ensure image is a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # Run YOLO model inference
    pred_bbox = model(input_1=image_tensor)

    # Extract first tensor output
    output_tensor = list(pred_bbox.values())[0]  # Extract YOLO raw output

    # Split tensor into bounding boxes, objectness scores, and class scores
    boxes, objectness, class_probs = tf.split(output_tensor, [4, 1, -1], axis=-1)

    # Apply Non-Maximum Suppression (NMS) for final detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(class_probs, (tf.shape(class_probs)[0], -1, tf.shape(class_probs)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )

    # Select the highest confidence detection
    valid_count = valid_detections.numpy()[0]
    if valid_count > 0:
        highest_idx = tf.argmax(scores[0][:valid_count])  # Index of highest score
        selected_class = classes[0][highest_idx].numpy()  # Most confident class
        selected_score = scores[0][highest_idx].numpy()  # Corresponding confidence
    else:
        selected_class = -1  # No valid detections
        selected_score = 0.0

    print(f"[INFO] Detected Class: {selected_class} with Confidence: {selected_score}")

    return selected_class, selected_score



def preprocess_image(image_path, input_size):
    """
    Load and preprocess the image for YOLO.

    :param image_path: Path to the input image
    :param input_size: Size to resize the image to (e.g., 416x416 for YOLOv4)
    :return: Preprocessed image as a NumPy array with shape (1, input_size, input_size, 3)
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"[ERROR] Image not found at path: {image_path}")

    # Convert BGR to RGB (YOLO models expect RGB input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to match YOLO input size
    image = cv2.resize(image, (input_size, input_size))

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    # Add batch dimension (YOLO expects batch input)
    return np.expand_dims(image.astype(np.float32), axis=0)


class YoloModelWrapper:
    def __init__(self, model_path):
        print(f"[INFO] Loading YOLOv4 model from: {model_path}")
        self.model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        self.infer = self.model.signatures['serving_default']
        print("[INFO] Model loaded successfully!")

    @tf.function
    def inference(self, x):
        output_dict = self.infer(input_1=x)  # Run inference
        output_tensor = list(output_dict.values())[0]  # Extract first output
        class_scores = output_tensor[..., 5:]  # Extract only class scores

        # Reduce across detected objects to get a final 2D logits tensor
        max_scores = tf.reduce_max(class_scores, axis=1)

        return max_scores  # Now it has shape (batch_size, num_classes)


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # Load YOLO model (Persistent Wrapper)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    yolo_model = YoloModelWrapper(FLAGS.weights)  # Use wrapper class

    # Preprocess input image
    input_size = FLAGS.size
    image = preprocess_image(FLAGS.image, input_size)

    # Convert image to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # Convert to EagerPy tensor (ensuring Foolbox compatibility)
    fimage = ep.astensor(image_tensor)

    # Extract YOLO class scores
    selected_class, selected_score = extract_yolo_detections(yolo_model.infer, image_tensor)

    print("[INFO] Extracted class:", selected_class)
    # Convert YOLO model to Foolbox-compatible model
    fmodel = fb.TensorFlowModel(yolo_model.inference, bounds=(0, 1))

    # Define and run Foolbox attack
    attack = fb.attacks.LinfPGD()
    
    # Convert labels to Foolbox-compatible EagerPy tensor
    labels_ep = ep.astensor(tf.convert_to_tensor([selected_class], dtype=tf.int64))

    print("[INFO] Label shape:",labels_ep.shape)
    # Pass correctly formatted inputs to Foolbox
    raw_advs, clipped_advs, success = attack(fmodel, fimage, criterion=fb.criteria.Misclassification(labels_ep), epsilons=0.03)

    print(f"[INFO] Attack success rate: {success.numpy().mean()}")

    # Convert adversarial image back to uint8 format (0-255 range)
    adv_image = (clipped_advs.numpy() * 255).astype(np.uint8)

    # Save adversarial image
    adv_image_path = "./image/adv_image.jpg"
    cv2.imwrite(adv_image_path, cv2.cvtColor(adv_image[0], cv2.COLOR_RGB2BGR))

    print(f"[INFO] Adversarial image saved at: {adv_image_path}")

if __name__ == '__main__':
    app.run(main)

 
