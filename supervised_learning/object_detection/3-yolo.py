#!/usr/bin/env python3
"""Task 3 - Non-max Suppression"""
import tensorflow as tf
import numpy as np


class Yolo:
    """
    Class to perform Non-max Suppression
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
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Function to filter boxes.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for box, conf, probs in zip(boxes, box_confidences, box_class_probs):
            scores = conf * probs
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)
            mask = best_score >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_score[mask])
        return (np.concatenate(filtered_boxes, axis=0),
                np.concatenate(box_classes, axis=0),
                np.concatenate(box_scores, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Function for non-max suppression.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Isolate all boxes for this class
            idx = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            # Sort by score descending
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while len(cls_boxes) > 0:
                # Always keep the highest-score box
                keep.append(0)

                if len(cls_boxes) == 1:
                    break

                # Compute IoU of best box vs all remaining
                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                area_best = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                             (cls_boxes[0, 3] - cls_boxes[0, 1]))
                areas_rest = ((cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                              (cls_boxes[1:, 3] - cls_boxes[1:, 1]))
                union = area_best + areas_rest - intersection

                iou = intersection / union

                # Keep only boxes with IoU below threshold
                low_iou = np.where(iou < self.nms_t)[0]
                cls_boxes = cls_boxes[low_iou + 1]
                cls_scores = cls_scores[low_iou + 1]

            box_predictions.append(cls_boxes[:len(keep)] if len(keep) < len(cls_boxes)
                                   else cls_boxes)
            predicted_box_classes.append(
                np.full(len(keep), cls, dtype=box_classes.dtype))
            predicted_box_scores.append(cls_scores[:len(keep)])

        return (np.concatenate(box_predictions, axis=0),
                np.concatenate(predicted_box_classes, axis=0),
                np.concatenate(predicted_box_scores, axis=0))
