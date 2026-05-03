#!/usr/bin/env python3
"""Task 7 - Predict"""
import tensorflow as tf
import numpy as np
import cv2
import os


class Yolo:
    """Performs object detection using the YOLOv3 algorithm."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Loads the Keras model, class names, and detection parameters."""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _sigmoid(self, x):
        """Applies sigmoid activation element-wise."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Decodes raw outputs into (x1, y1, x2, y2) pixel coordinates."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            cx = np.tile(
                np.arange(grid_w).reshape(1, grid_w, 1),
                (grid_h, 1, anchor_boxes)
            )
            cy = np.tile(
                np.arange(grid_h).reshape(grid_h, 1, 1),
                (1, grid_w, anchor_boxes)
            )

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = (self._sigmoid(output[..., 0]) + cx) / grid_w
            by = (self._sigmoid(output[..., 1]) + cy) / grid_h
            bw = (pw * np.exp(output[..., 2])) / input_w
            bh = (ph * np.exp(output[..., 3])) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            box_confidences.append(self._sigmoid(output[..., 4:5]))
            box_class_probs.append(self._sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Keeps boxes whose best class score meets the confidence threshold."""
        fb, bc, bs = [], [], []

        for box, conf, probs in zip(boxes, box_confidences, box_class_probs):
            scores = conf * probs
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)
            mask = best_score >= self.class_t
            fb.append(box[mask])
            bc.append(best_class[mask])
            bs.append(best_score[mask])

        return (np.concatenate(fb, axis=0),
                np.concatenate(bc, axis=0),
                np.concatenate(bs, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Suppresses redundant overlapping boxes per class."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]
            cb = filtered_boxes[idx]
            cs = box_scores[idx]

            order = np.argsort(cs)[::-1]
            cb = cb[order]
            cs = cs[order]

            keep_boxes = []
            keep_scores = []

            while len(cb) > 0:
                keep_boxes.append(cb[0])
                keep_scores.append(cs[0])

                if len(cb) == 1:
                    break

                x1 = np.maximum(cb[0, 0], cb[1:, 0])
                y1 = np.maximum(cb[0, 1], cb[1:, 1])
                x2 = np.minimum(cb[0, 2], cb[1:, 2])
                y2 = np.minimum(cb[0, 3], cb[1:, 3])

                inter = (
                    np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                )
                area_best = (
                    (cb[0, 2] - cb[0, 0]) * (cb[0, 3] - cb[0, 1])
                )
                areas_rest = (
                    (cb[1:, 2] - cb[1:, 0]) * (cb[1:, 3] - cb[1:, 1])
                )
                union = area_best + areas_rest - inter
                iou = inter / union

                keep_idx = np.where(iou < self.nms_t)[0]
                cb = cb[keep_idx + 1]
                cs = cs[keep_idx + 1]

            box_predictions.append(np.array(keep_boxes))
            predicted_box_classes.append(
                np.full(len(keep_boxes), cls, dtype=box_classes.dtype)
            )
            predicted_box_scores.append(np.array(keep_scores))

        return (np.concatenate(box_predictions, axis=0),
                np.concatenate(predicted_box_classes, axis=0),
                np.concatenate(predicted_box_scores, axis=0))

    @staticmethod
    def load_images(folder_path):
        """Loads all images from a folder into a list of numpy arrays."""
        images = []
        image_paths = []
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(extensions):
                path = os.path.join(folder_path, fname)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Resizes and normalizes images to match the model's input format."""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])
            resized = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Draws boxes and labels on the image and handles save on 's' key."""
        img = image.copy()

        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = "{}: {:.2f}".format(self.class_names[cls], score)
            cv2.putText(
                img, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, img)
        key = cv2.waitKey(0)

        if key == ord('s'):
            os.makedirs('detections', exist_ok=True)
            cv2.imwrite(os.path.join('detections', file_name), img)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Runs the full detection pipeline on all images in a folder."""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        # Run all images through model in one batch
        outputs = self.model.predict(pimages)

        if not isinstance(outputs, list):
            outputs = [outputs]

        predictions = []

        for i, img in enumerate(images):
            # Slice this image's output across all 3 scales
            img_outputs = [out[i] for out in outputs]

            boxes, box_confidences, box_class_probs = (
                self.process_outputs(img_outputs, image_shapes[i])
            )
            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores
            )

            # Window name is filename only, no directory path
            file_name = os.path.basename(image_paths[i])
            self.show_boxes(img, boxes, box_classes, box_scores, file_name)

            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths
