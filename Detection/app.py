import os
import base64
import torch.cuda
from flask import Flask, render_template, request, g, jsonify
import detection

app = Flask(__name__)


def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """

    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        threshold = request.form.get('threshold', type=float)
        if not isinstance(threshold, float):
            threshold = 0.5

        data_dir = r'sources/cache'
        # 通过表单中name值获取图片
        imgData = request.files["img"]
        # 获取图片名称及后缀名
        imgName = imgData.filename
        # 保存图片
        data_path = os.path.join(data_dir, imgName)
        imgData.save(data_path)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # 目标检测
        save_dir_detection = r'sources/detections'
        saved_img_path_detection = detection.model(data_path, save_dir_detection, threshold=threshold,
                                                   device=device).forward()

        img_stream_detection = return_img_stream(saved_img_path_detection)

        return render_template('index.html', img_stream_detection=img_stream_detection)

if __name__ == '__main__':
    app.run()
