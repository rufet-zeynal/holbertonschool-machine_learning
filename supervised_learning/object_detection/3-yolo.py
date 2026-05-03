#!/usr/bin/env python3
"""Task 3 - Non-max Suppression"""
import tensorflow as tf
import numpy as np


class Yolo:
    """Performs object detection using the YOLOv3 algorithm."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Loads the Keras model, class names, and detection parameters.
        """
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
        """
        Decodes raw outputs into (x1, y1, x2, y2) pixel coordinates.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Grid offsets so each cell knows its (col, row) position
            cx = np.tile(
                np.arange(grid_w).reshape(1, grid_w, 1),
                (grid_h, 1, anchor_boxes)
            )
            cy = np.tile(
                np.arange(grid_h).reshape(grid_h, 1, 1),
                (1, grid_w, anchor_boxes)
            )

            # Anchor dimensions for this scale
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Decode box center as fraction of full image
            bx = (self._sigmoid(output[..., 0]) + cx) / grid_w
            by = (self._sigmoid(output[..., 1]) + cy) / grid_h

            # Decode box size as fraction of input dimensions
            bw = (pw * np.exp(output[..., 2])) / input_w
            bh = (ph * np.exp(output[..., 3])) / input_h

            # Convert center+size to corner pixel coords in original image
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Sigmoid-activate confidence and class probabilities
            box_confidences.append(self._sigmoid(output[..., 4:5]))
            box_class_probs.append(self._sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Keeps boxes whose best class score meets the confidence threshold.
        """
        fb, bc, bs = [], [], []

        for box, conf, probs in zip(boxes, box_confidences, box_class_probs):
            # Score = objectness confidence * class probability
            scores = conf * probs

            # Pick the class with the highest score for each box
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)

            # Keep only boxes whose best score clears the threshold
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

        # Process each class independently so different classes never
        # suppress each other
        for cls in np.unique(box_classes):

            # Isolate all boxes belonging to this class
            idx = np.where(box_classes == cls)[0]
            cb = filtered_boxes[idx]
            cs = box_scores[idx]

            # Sort descending so the best box is always at index 0
            order = np.argsort(cs)[::-1]
            cb = cb[order]
            cs = cs[order]

            keep_boxes = []
            keep_scores = []

            while len(cb) > 0:
                # Keep the current best box
                keep_boxes.append(cb[0])
                keep_scores.append(cs[0])

                if len(cb) == 1:
                    break

                # Intersection corners: best box vs all remaining
                x1 = np.maximum(cb[0, 0], cb[1:, 0])
                y1 = np.maximum(cb[0, 1], cb[1:, 1])
                x2 = np.minimum(cb[0, 2], cb[1:, 2])
                y2 = np.minimum(cb[0, 3], cb[1:, 3])

                # Intersection area, clamped to 0 for non-overlapping boxes
                inter = (
                    np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                )

                # Individual areas
                area_best = (
                    (cb[0, 2] - cb[0, 0]) * (cb[0, 3] - cb[0, 1])
                )
                areas_rest = (
                    (cb[1:, 2] - cb[1:, 0]) * (cb[1:, 3] - cb[1:, 1])
                )

                # Union = sum of both areas minus shared intersection
                union = area_best + areas_rest - inter

                # IoU: 0 = no overlap, 1 = identical boxes
                iou = inter / union

                # Discard boxes that overlap too much — same object
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
