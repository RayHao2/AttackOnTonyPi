import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

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
    
    return infer


def pgd_attack_yolo_tf(model, image_path, input_size, base_class_id, target_class_id, epsilon=0.03, alpha=0.01, iterations=40):
    """
    Perform Projected Gradient Descent (PGD) attack on YOLOv4 model.

    :param model: YOLO model inference function
    :param image_path: Path to the input image
    :param input_size: YOLO model input size (e.g., 416)
    :param base_class_id: The current class ID to attack (e.g., dog)
    :param target_class_id: The desired misclassification class ID (e.g., person)
    :param epsilon: Maximum perturbation magnitude
    :param alpha: Step size for each iteration
    :param iterations: Number of attack iterations
    :return: Adversarial image
    """

    # 1️⃣ **Load & Preprocess Image**
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size)) / 255.0  # Normalize to [0,1]

    # Expand dimensions to create batch (YOLO expects batch input)
    images_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # Convert image data to TensorFlow tensor
    images_data_tf = tf.convert_to_tensor(images_data, dtype=tf.float32)

    # 2️⃣ **Initialize adversarial perturbation δ within [-ε, ε]**
    delta = tf.Variable(tf.random.uniform(shape=images_data_tf.shape, minval=-epsilon, maxval=epsilon, dtype=tf.float32))

    # 3️⃣ **PGD Attack Loop**
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(delta)

            # Create adversarial image (x + δ)
            adv_image = images_data_tf + delta
            adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)  # Keep pixel values in range [0,1]

            # ✅ Pass input correctly (use `input_1=adv_image` instead of positional arguments)
            pred_bbox = model(input_1=adv_image)

            # Extract class predictions
            for key, value in pred_bbox.items():
                pred_classes = value[:, :, 5:]  # Extract class probabilities

            # **Compute loss** (maximize probability of the **wrong** class)
            loss = -tf.reduce_mean(pred_classes[:, :, target_class_id])  # Increase target class confidence

        # Compute gradient of loss w.r.t δ
        gradient = tape.gradient(loss, delta)
        signed_grad = tf.sign(gradient)  # Get sign of gradient

        # **Update perturbation**
        delta.assign_add(alpha * signed_grad)

        # **Project δ to keep it within [-ε, ε]**
        delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))

        print(f"[DEBUG] Iteration {i+1}/{iterations}: Loss = {loss.numpy()}")

    # 4️⃣ **Create Final Adversarial Image**
    adv_image = tf.clip_by_value(images_data_tf + delta, 0.0, 1.0)

    return adv_image.numpy()  # Convert back to NumPy array for saving



def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    # Load YOLO model
    infer = load_yolo_model(FLAGS.weights)

    # Load and preprocess image
    input_size = FLAGS.size
    image_path = FLAGS.image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = np.asarray([image_data]).astype(np.float32)

    # Run PGD attack on image
    base_class_id = 16  # Example: Dog (Modify based on image content)
    target_class_id = 0  # Example: Person (Misclassification target)

    print("[INFO] Running PGD attack...")
    adv_image = pgd_attack_yolo_tf(infer, image_path, input_size, base_class_id, target_class_id)

    # Save adversarial image
    adv_image_uint8 = (adv_image[0] * 255).astype(np.uint8)  # Convert to uint8
    adv_image_path = "./image/adv_image.png"
    cv2.imwrite(adv_image_path, cv2.cvtColor(adv_image_uint8, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Adversarial image saved at: {adv_image_path}")

    # Run YOLO on adversarial image
    print("[INFO] Running YOLOv4 on adversarial image...")
    batch_data = tf.constant(adv_image)
    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # Perform non-max suppression
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    print("[INFO] YOLO predictions on adversarial image:")
    print("[DEBUG] Classes:", classes.numpy())
    print("[DEBUG] Scores:", scores.numpy())
    print("[DEBUG] Bounding Boxes:", boxes.numpy())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
