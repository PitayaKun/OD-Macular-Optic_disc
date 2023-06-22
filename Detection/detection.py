import imageio
import torch
from imgaug import BoundingBoxesOnImage, BoundingBox
import utils
import os

from models.detection import myfaster_rcnn, roihead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def resolve(detection, threshold=0.5):
    assert isinstance(detection, dict)
    boxes, labels, scores = detection['boxes'].tolist(), detection['labels'].tolist(), detection['scores'].tolist()
    macular_box, opticDisc_box = [], []
    macular_flag, opticDisc_flag = True, True
    for box, label, score in zip(boxes, labels, scores):
        if macular_flag and label == 1 and score > threshold:
            macular_box = box
            macular_flag = False
        elif opticDisc_flag and label == 2 and score > threshold:
            opticDisc_box = box
            opticDisc_flag = False
    return macular_box, opticDisc_box


def draw(img_path, macular_box, opticDisc_box, save_path):
    image = imageio.imread(img_path)
    axis1 = macular_box if len(macular_box) else [0, 0, 0, 0]
    axis2 = opticDisc_box if len(opticDisc_box) else [0, 0, 0, 0]
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=axis1[0], x2=axis1[2], y1=axis1[1], y2=axis1[3]),
        BoundingBox(x1=axis2[0], x2=axis2[2], y1=axis2[1], y2=axis2[3])
    ], shape=image.shape)
    bbs_labeled = bbs.deepcopy()
    bbs_labeled[0].label = "macular"
    bbs_labeled[1].label = "opticDisc"
    image_bbs = bbs_labeled.draw_on_image(image, size=2)
    # image_bbs = bbs.bounding_boxes[0].draw_on_image(image_bbs, color=[0, 255, 0], size=3)
    # image_bbs = bbs.bounding_boxes[1].draw_on_image(image_bbs, color=[255, 0, 0], size=3)
    imageio.imwrite(save_path, image_bbs)


class model:
    def __init__(self, data_path, save_dir, threshold=0.5, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=3)

        self.net = myfaster_rcnn.MyFasterRCNN(backbone, num_classes=3,
                                image_mean=[0.46214035, 0.3716353, 0.38595816],
                                image_std=[0.18190928, 0.1311566, 0.11936087]).to(device)
        state_dict = torch.load(r'./models/resnet18.pth')
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.data_path = data_path
        self.save_dir = save_dir
        self.threshold = threshold

    def forward(self):
        loader = utils.image_loader(self.data_path, self.device)
        saved_img_paths = []
        for img_tensor, img_path in loader:
            x = [img_tensor]
            with torch.no_grad():
                detection = self.net(x)[0]
            macular_box, opticDisc_box = resolve(detection, threshold=self.threshold)
            _, img_name = os.path.split(img_path)
            save_path = os.path.join(self.save_dir, img_name)
            draw(img_path, macular_box, opticDisc_box, save_path)
            saved_img_paths.append(save_path)
        return saved_img_paths if len(saved_img_paths) != 1 else saved_img_paths[0]
