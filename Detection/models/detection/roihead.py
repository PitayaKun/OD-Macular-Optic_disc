import torch
import torchvision.models.detection.roi_heads as rh
import torch.nn.functional as F
from typing import List, Dict


class MyRoIHeads(rh.RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                 positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img):
        super(MyRoIHeads, self).__init__(
            box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img)
        self.batch_size_per_image = batch_size_per_image

    def distance_loss(self, class_logits, box_regression):
        N, num_classes = class_logits.shape
        batch_size = N // self.batch_size_per_image
        class_logits = class_logits.reshape(batch_size, self.batch_size_per_image, num_classes)
        pred_scores = F.softmax(class_logits, -1)
        box_regression = box_regression.reshape(batch_size, self.batch_size_per_image, num_classes, 4)
        scores, indices = torch.max(pred_scores, dim=1)
        loss_distance = 0
        counter = 0
        for i in range(batch_size):
            if scores[i][1] < 0.5 or scores[i][2] < 0.5:
                continue
            macular_box = box_regression[i, indices[i][1], 1]
            opticdisc_box = box_regression[i, indices[i][2], 2]
            macularx = (macular_box[0] + macular_box[2]) / 2
            maculary = (macular_box[1] + macular_box[3]) / 2
            opticdiscx = (opticdisc_box[0] + opticdisc_box[2]) / 2
            opticdiscy = (opticdisc_box[1] + opticdisc_box[3]) / 2
            distance = ((macularx - opticdiscx) ** 2 + (maculary - opticdiscy) ** 2) ** 0.5 * 3
            if distance < 650:
                loss_distance += torch.exp((650 - distance) / 1000) - 1
            elif distance > 950:
                loss_distance += torch.exp((distance - 950) / 1000) - 1
            else:
                loss_distance += 0
            counter += 1
        return loss_distance / counter if counter != 0 else 0

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        fc_class_logits, fc_box_regression, conv_class_logits, conv_box_regression = self.box_predictor(box_features)
        # 分类融合
        # complementary_class_logits = fc_class_logits + conv_class_logits * (1 - fc_class_logits)
        complementary_class_logits = fc_class_logits
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            fc_loss_classifier, fc_loss_box_reg = rh.fastrcnn_loss(
                fc_class_logits, fc_box_regression, labels, regression_targets)
            conv_loss_classifier, conv_loss_box_reg = rh.fastrcnn_loss(
                conv_class_logits, conv_box_regression, labels, regression_targets)

            # lambda_fc, lambda_conv 按照论文分别使用0.7，0.8
            lambda_fc, lambda_conv = 0.7, 0.8
            loss_fc = lambda_fc * fc_loss_classifier + (1 - lambda_fc) * fc_loss_box_reg
            loss_conv = (1 - lambda_conv) * conv_loss_classifier + lambda_conv * conv_loss_box_reg

            box_regression = self.box_coder.decode(conv_box_regression, proposals)
            # 计算距离损失
            loss_distance = self.distance_loss(complementary_class_logits, box_regression)
            losses = {
                "loss_fc": loss_fc,
                "loss_conv": loss_conv,
                "loss_distance": loss_distance
            }
        else:
            # 分类融合fc-head和conv-head的分类分数，回归框回归使用conv-head的结果
            boxes, scores, labels = \
                self.postprocess_detections(complementary_class_logits, conv_box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result, losses
