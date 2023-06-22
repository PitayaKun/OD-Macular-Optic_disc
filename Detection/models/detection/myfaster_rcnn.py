import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from .roihead import MyRoIHeads


class MyFasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = DoubleHead(
                out_channels, resolution,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = DoubleHeadPredictor(
                representation_size,
                num_classes)

        roi_heads = MyRoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(MyFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class UpSampleBlock(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1
                 ) -> None:
        super(UpSampleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv_identity = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn_identity = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = self.conv_identity(x)
        identity = self.bn_identity(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class NonlocalBlock(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1
                 ) -> None:
        super(NonlocalBlock, self).__init__()
        width = inplanes // 2
        norm_layer = nn.BatchNorm2d
        self.conv1x1_theta = \
            nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(width)
        self.conv1x1_phi = \
            nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv1x1_g = \
            nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn3 = norm_layer(width)
        self.conv1x1_zeta = \
            nn.Conv2d(width, planes, kernel_size=1, stride=stride, groups=groups, bias=False, dilation=dilation)
        self.bn4 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        theta = self.conv1x1_theta(x)
        theta = self.bn1(theta)
        theta = self.relu(theta)

        phi = self.conv1x1_phi(x)
        phi = self.bn2(phi)
        phi = self.relu(phi)

        g = self.conv1x1_g(x)
        g = self.bn3(g)
        g = self.relu(g)

        theta = theta.flatten(start_dim=2).transpose(1, 2).contiguous()
        phi = phi.flatten(start_dim=2)
        out = torch.matmul(theta, phi)

        g = g.flatten(start_dim=2).transpose(1, 2).contiguous()
        out = torch.matmul(out, g)
        out = out.reshape(n, -1, h, w)

        out = self.conv1x1_zeta(out)
        out = self.bn4(out)

        out += identity
        out = self.relu(out)

        return out


class DoubleHead(nn.Module):
    def __init__(self, in_channels, resolution, representation_size):
        super(DoubleHead, self).__init__()
        # fc-head
        self.fc6 = nn.Linear(in_channels * resolution**2, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        # conv-head
        self.conv1 = UpSampleBlock(in_channels, representation_size)
        self.nlblock1 = NonlocalBlock(representation_size, representation_size)
        self.conv2 = resnet.Bottleneck(representation_size, representation_size // 4)
        self.nlblock2 = NonlocalBlock(representation_size, representation_size)
        self.conv3 = resnet.Bottleneck(representation_size, representation_size // 4)

    def forward(self, x):
        # output of fc-head
        fc_out = x.flatten(start_dim=1)
        fc_out = F.relu(self.fc6(fc_out))
        fc_out = F.relu(self.fc7(fc_out))

        # output of conv-head
        conv_out = self.conv1(x)
        conv_out = self.nlblock1(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.nlblock2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = F.adaptive_avg_pool2d(conv_out, 1).squeeze()

        return fc_out, conv_out


class DoubleHeadPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DoubleHeadPredictor, self).__init__()
        self.fc_cls_score = nn.Linear(in_channels, num_classes)
        self.fc_bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.conv_cls_score = nn.Linear(in_channels, num_classes)
        self.conv_bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        fc_out, conv_out = x

        fc_scores = self.fc_cls_score(fc_out)
        fc_bbox_deltas = self.fc_bbox_pred(fc_out)
        conv_scores = self.conv_cls_score(conv_out)
        conv_bbox_deltas = self.conv_bbox_pred(conv_out)

        return fc_scores, fc_bbox_deltas, conv_scores, conv_bbox_deltas
