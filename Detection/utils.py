import json
import pickle
import random
import shutil
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import os


def image_loader(data_path, device=torch.device('cpu')):
    if os.path.isdir(data_path):
        for dir_file_name in os.listdir(data_path):
            dir_file_path = os.path.join(data_path, dir_file_name)
            for img_tensor, img_path in image_loader(dir_file_path, device):
                yield img_tensor, img_path
    elif data_path.endswith('.jpg'):
        img_tensor = default_loader(data_path).to(device)
        yield img_tensor, data_path


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def iou(box, target_box):
    xA = max(box[0], target_box[0])
    yA = max(box[1], target_box[1])
    xB = min(box[2], target_box[2])
    yB = min(box[3], target_box[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    target_box_area = (target_box[2] - target_box[0] + 1) * (target_box[3] - target_box[1] + 1)

    value_iou = inter_area / float(box_area + target_box_area - inter_area)
    return value_iou


def acc(boxes, target_boxes):
    # 计算iou和不计算iou的准确率
    iou_list = []
    right, total = 0, 0

    for box, target_box in zip(boxes, target_boxes):
        total += 1
        if len(box) and len(target_box):
            iou_list.append(iou(box, target_box))
        if len(box) == len(target_box):
            right += 1

    print(np.array(iou_list).mean())
    accuracy = right / total
    # 计算用于AP的精确率和召回率列表
    p_list = []
    r_list = []
    for threshold in set(iou_list):
        tp, fp, fn = 0, 0, 0
        iou_index = 0
        for box, target_box in zip(boxes, target_boxes):
            if len(box) and len(target_box):
                if iou_list[iou_index] >= threshold:
                    tp += 1
                else:
                    fn += 1
                iou_index += 1
            elif len(box) or len(target_box):
                if len(box):
                    fp += 1
                else:
                    fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0
        p_list.append(precision)
        recall = tp / (tp + fn) if (tp + fn) else 0
        r_list.append(recall)
    # 计算AP
    thresholds = r_list + [0, 1]
    thresholds.sort()
    AP = 0
    last_threshold = 0
    for threshold in thresholds:
        max_p = 0
        for p, r in zip(p_list, r_list):
            if r >= threshold and p > max_p:
                max_p = p
        AP += (threshold - last_threshold) * max_p
        last_threshold = threshold
    return {'AP': AP, 'acc': accuracy}


def resolve(detections, targets, mode=0):
    """
        mode=0:黄斑和视盘
        mode=1：仅黄斑
        mode=2：仅视盘
    """
    macular_boxes, opticdisc_boxes = [], []
    target_macular_boxes, target_opticdisc_boxes = [], []
    # 模式0
    if mode == 0:
        for detection, target in zip(detections, targets):
            # 整理detection
            macular_box, opticdisc_box = [], []
            macular_flag, opticdisc_flag = True, True
            boxes = detection['boxes'].tolist()
            assert isinstance(boxes, list)
            for index, (label, score) in enumerate(zip(detection['labels'].tolist(), detection['scores'].tolist())):
                if macular_flag and label == 1 and score >= 0.4:
                    macular_box = boxes[index]
                    macular_flag = False
                if opticdisc_flag and label == 2 and score >= 0.4:
                    opticdisc_box = boxes[index]
                    opticdisc_flag = False
            macular_boxes.append(macular_box)
            opticdisc_boxes.append(opticdisc_box)
            # 整理target
            target_macular_box, target_opticdisc_box = [], []
            target_macular_flag, target_opticdisc_flag = True, True
            target_boxes = target['boxes'].tolist()
            assert isinstance(target_boxes, list)
            for index, label in enumerate(target['labels'].tolist()):
                if target_macular_flag and label == 1:
                    target_macular_box = target_boxes[index]
                    target_macular_flag = False
                if target_opticdisc_flag and label == 2:
                    target_opticdisc_box = target_boxes[index]
                    target_opticdisc_flag = False
            target_macular_boxes.append(target_macular_box)
            target_opticdisc_boxes.append(target_opticdisc_box)
        return macular_boxes, target_macular_boxes, opticdisc_boxes, target_opticdisc_boxes
    # 模式1
    elif mode == 1:
        for detection, target in zip(detections, targets):
            # 整理detection
            macular_box = []
            macular_flag = True
            boxes = detection['boxes'].tolist()
            assert isinstance(boxes, list)
            for index, (label, _) in enumerate(zip(detection['labels'].tolist(), detection['scores'].tolist())):
                if macular_flag and label == 1:
                    macular_box = boxes[index]
                    macular_flag = False
            macular_boxes.append(macular_box)
            # 整理target
            target_macular_box = []
            target_macular_flag = True
            target_boxes = target['boxes'].tolist()
            assert isinstance(target_boxes, list)
            for index, label in enumerate(target['labels'].tolist()):
                if target_macular_flag and label == 1:
                    target_macular_box = target_boxes[index]
                    target_macular_flag = False
            target_macular_boxes.append(target_macular_box)
        return macular_boxes, target_macular_boxes
    # 模式2
    elif mode == 2:
        for detection, target in zip(detections, targets):
            # 整理detection
            opticdisc_box = []
            opticdisc_flag = True
            boxes = detection['boxes'].tolist()
            assert isinstance(boxes, list)
            for index, (label, _) in enumerate(zip(detection['labels'].tolist(), detection['scores'].tolist())):
                if opticdisc_flag and label == 2:
                    opticdisc_box = boxes[index]
                    opticdisc_flag = False
            opticdisc_boxes.append(opticdisc_box)
            # 整理target
            target_opticdisc_box = []
            target_opticdisc_flag = True
            target_boxes = target['boxes'].tolist()
            assert isinstance(target_boxes, list)
            for index, label in enumerate(target['labels'].tolist()):
                if target_opticdisc_flag and label == 2:
                    target_opticdisc_box = target_boxes[index]
                    target_opticdisc_flag = False
            target_opticdisc_boxes.append(target_opticdisc_box)
        return opticdisc_boxes, target_opticdisc_boxes


def create_dataset(dataset_path, save_path, num_images=1000, proportion=0.2):
    image_names = []
    for image_name in os.listdir(dataset_path):
        image_names.append(image_name)
    random_indices = random.sample(range(len(image_names)), num_images)
    counter = 0
    threshold = num_images * proportion
    trainset_path = os.path.join(save_path, 'train')
    if not os.path.exists(trainset_path):
        os.mkdir(trainset_path)
    testset_path = os.path.join(save_path, 'test')
    if not os.path.exists(testset_path):
        os.mkdir(testset_path)
    for index in random_indices:
        save_path = testset_path if counter < threshold else trainset_path
        shutil.move(os.path.join(dataset_path, image_names[index]), save_path)
        # shutil.copy(os.path.join(dataset_path, image_names[index]), save_path)
        counter += 1


def axis_list2str(axis_list):
    return '[' + str(axis_list[0]) + ', ' + str(axis_list[1]) + ', ' \
           + str(axis_list[2]) + ', ' + str(axis_list[3]) + ']'


def update_csv(dataset_path, csv_path):
    # 读取csv
    df = pd.read_csv(csv_path)
    # 不存在则新增列
    if 'opticDisc' not in df.columns:
        df['opticDisc'] = np.nan
    for dir_file in os.listdir(dataset_path):
        dir_file_path = os.path.join(dataset_path, dir_file)
        if os.path.isdir(dir_file_path):
            df = update_csv(dir_file_path, csv_path)
            df.to_csv(csv_path, index=False)
        else:
            if dir_file_path.endswith('.json'):
                with open(dir_file_path, 'r', encoding='utf-8') as json_str:
                    json_dict = json.load(json_str)
                if len(json_dict['shapes']) != 0:
                    opticdisc_dict = json_dict['shapes'][0]
                    if opticdisc_dict['label'] == '1':
                        # find the opticDisc axis
                        points = opticdisc_dict['points']
                        x1, y1, x2, y2 = points[0][0], points[0][1], points[1][0], points[1][1]

                        if opticdisc_dict['shape_type'] == 'circle':
                            r = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                            cx, cy = x1, y1
                            x1 = cx - r
                            y1 = cy - r
                            x2 = cx + r
                            y2 = cy + r
                        opticdisc_axis = \
                            [x1 if x1 >= 0 else 0, y1 if y1 >= 0 else 0, x2 if x2 >= 0 else 0, y2 if y2 >= 0 else 0]
                        img_filename = json_dict['imagePath']
                        df.loc[df['filename'] == img_filename, 'opticDisc'] = axis_list2str(opticdisc_axis)
    return df


def default_loader(path):
    # print(path)
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    img_pil = Image.open(path).convert('RGB')
    img_tensor = preprocess(img_pil)
    return img_tensor


class dataset(Dataset):
    def __init__(self, dataset_path, csv_path, device, train=True, loader=default_loader, mode=0):
        """
            mode=0:黄斑和视盘
            mode=1：仅黄斑
            mode=2：仅视盘
        """
        df = pd.read_csv(csv_path)
        images = []
        targets = []
        if train:
            self.dataset_path = os.path.join(dataset_path, 'train')
            self.mean_std_path = r'mean_std_value_train.pkl'
        else:
            self.dataset_path = os.path.join(dataset_path, 'test')
            self.mean_std_path = r'mean_std_value_test.pkl'
        for file_name in os.listdir(self.dataset_path):
            if file_name.endswith('.jpg'):
                images_name = file_name
                images_path = os.path.join(self.dataset_path, images_name)
                # create the dict according to the csv
                boxes = []
                labels = []
                label_dict = df.loc[df['filename'] == images_name].to_dict(orient='records')[0]
                # add image
                images.append(images_path)
                if mode == 0:
                    if not pd.isnull(label_dict['macular']):
                        assert isinstance(label_dict['macular'], str)
                        macular_str = label_dict['macular']
                        box = [float(num) for num in macular_str.strip('[').strip(']').split(', ')]
                        boxes.append(box)
                        labels.append(1)
                    if not pd.isnull(label_dict['opticDisc']):
                        assert isinstance(label_dict['opticDisc'], str)
                        opticdisc_str = label_dict['opticDisc']
                        box = [float(num) for num in opticdisc_str.strip('[').strip(']').split(', ')]
                        boxes.append(box)
                        labels.append(2)
                elif mode == 1:
                    if not pd.isnull(label_dict['macular']):
                        assert isinstance(label_dict['macular'], str)
                        macular_str = label_dict['macular']
                        box = [float(num) for num in macular_str.strip('[').strip(']').split(', ')]
                        boxes.append(box)
                        labels.append(1)
                elif mode == 2:
                    if not pd.isnull(label_dict['opticDisc']):
                        assert isinstance(label_dict['opticDisc'], str)
                        macular_str = label_dict['opticDisc']
                        box = [float(num) for num in macular_str.strip('[').strip(']').split(', ')]
                        boxes.append(box)
                        labels.append(1)
                target_dict = {
                    'boxes':
                        torch.tensor(boxes).to(device)
                        if len(boxes) != 0 else
                        torch.tensor(boxes).reshape(0, 4).to(device),
                    'labels': torch.as_tensor(labels, dtype=torch.int64).to(device)
                }
                targets.append(target_dict)
        self.images = images
        self.target = targets
        self.loader = loader
        self.means, self.stds = self.get_mean_std()
        self.device = device

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn).to(self.device)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)

    def get_mean_std(self):
        means = [0, 0, 0]
        stds = [0, 0, 0]
        if os.path.exists(self.mean_std_path):
            with open(self.mean_std_path, 'rb') as f:
                means = pickle.load(f)
                stds = pickle.load(f)
                print('pickle load done')
                return means, stds

        num_imgs = len(self.images)
        for images_path in self.images:
            img = default_loader(images_path)
            for i in range(3):
                # 一个通道的均值和标准差
                means[i] += img[i, :, :].mean()
                stds[i] += img[i, :, :].std()

        means = np.asarray(means) / num_imgs
        stds = np.asarray(stds) / num_imgs

        print("normMean = {}".format(means))
        print("normStds = {}".format(stds))

        # 将得到的均值和标准差写到文件中，之后就能够从中读取
        with open(self.mean_std_path, 'wb') as f:
            pickle.dump(means, f)
            pickle.dump(stds, f)
            print('pickle done')
        return means, stds
