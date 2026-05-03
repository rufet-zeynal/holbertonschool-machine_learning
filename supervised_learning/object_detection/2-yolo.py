#!/usr/bin/env python3
"""Task 2 - Filter Boxes"""
import tensorflow as tf
import numpy as np


class Yolo:
    """
    Task 2 - Filter Boxes
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Function to initialize the model.
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Function to process the output of the model.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]
        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)
            cy = np.tile(cy, (1, grid_w, anchor_boxes))
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_h
            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Function to filter boxes.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, probs in zip(boxes, box_confidences, box_class_probs):
            # box_score = confidence * class_probability
            scores = conf * probs                          # (..., classes)
            best_class = np.argmax(scores, axis=-1)        # (...,)
            best_score = np.max(scores, axis=-1)           # (...,)

            # Keep only boxes above threshold
            mask = best_score >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
