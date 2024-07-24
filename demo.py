import os
import shutil
import glob
import cv2
import torch
import time
import requests
import numpy as np
from tqdm import tqdm

from ucs_alg_node import AlgNode, Alg, AlgResult, AlgSubmitter, AlgNodeWeb, AlgTask, utils

from model import Fcn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Fcn()  # 定义模型
weights_path = "./weights/model_12.pth"  # 权重
weights = torch.load(weights_path,map_location='cpu')
model.load_state_dict(weights)
model.to(device)
save_folder = 'test/videos' #定义保存路径

task_id = 'task112'
node_id = 'node001'
alg_name = 'my_alg'
out_topic = 'ucs/alg/result'
file_path = 'D:/gugol/Motion_Emotion_Dataset'


class MyAlg(Alg):
    def __init__(self, mode, sources, model, id):
        super().__init__(mode, model)
        self.name = 'my_alg'
        self.sources = sources
        self.id = id

    def infer_batch(self, task):
        # file_path = self.sources  # 假设 self.sources 是视频文件所在的目录
        sources = 'D:/gugol/Motion_Emotion_Dataset'
        video_files = glob.glob(os.path.join(sources, '*.mp4'))

        all_labels = []
        for video_file in video_files:
            labels = self.process_video(video_file)
            all_labels.append({
                "video": video_file,
                "labels": labels
            })
            time.sleep(10)  # 模拟延迟

        ts = time.time_ns()
        result = AlgResult(task.id, ts, all_labels, "batch result")
        return result

    def process_video(self, video_file):
        # 读取本地视频文件
        cap = cv2.VideoCapture(video_file)

        ret, img = cap.read()  # 取出
        if not ret:
            print("读取视频失败: " + video_file)
            cap.release()
            cv2.destroyAllWindows()
            return []

        img_model = cv2.resize(img, (512, 512))  # 输入模型的第一帧图片，缩放
        img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)  # 二值
        prev_img_grey = img_grey  # 第一帧视频
        frame = 1  # 第二帧视频

        labels = []

        while True:
            ret, img = cap.read()  # 取出视频， img是原始视频

            if not ret:
                # print("读取视频结束")
                break

            img_model = cv2.resize(img, (512, 512))  # 输入视频
            img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_img_grey, img_grey, None, 0.5, 5, 15, 3, 5, 1.1,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            flow = flow[np.newaxis, :]
            flow = flow.transpose(0, 3, 1, 2)
            flow = torch.from_numpy(flow).float().to(model.device)

            predict = model(flow)  # 预测输出
            predict = torch.argmax(predict, 1).cpu().numpy()[0]

            labels.append(predict)

            frame += 1  #
            prev_img_grey = img_grey

        cap.release()
        cv2.destroyAllWindows()

        return labels



def main():
    cfg = {
        'id': node_id,
        'name': 'alg_name',
        'mode': 'batch',
        'max_task': 10,
        'model_dir': os.path.join(utils.get_cwd(), "model"),  # could be file path or url or model name
        'alg_id': 'alg_id123', # only effective in batch mode
        'web_port':9996
    }

    task = AlgTask(id=task_id,
                   ts=utils.current_time_milli(),
                   sources=['D:/gugol/Motion_Emotion_Dataset']
                   # sources= ['rtsp://localhost:9111/123',
                   #           'mqx://localhost:8011/1123']
    )

    alg_cfg = {
        # only effective in stream mode
        'alg_id': 'alg_id123',
        # only effective in stream mode
        'model': 'model_12.pth',  # could be file path or url or model name
    }

    out_cfg = {
        'dest': '62.234.16.239:1883',
        'mode': 'mqtt',
        'username': 'admin',
        'passwd': 'admin1234',
        'topic': out_topic
    }

    alg = MyAlg(cfg['mode'], task.sources, alg_cfg['model'], alg_cfg['alg_id'])

    submitter = AlgSubmitter(
        dest=out_cfg['dest'],
        mode=out_cfg['mode'],
        username=out_cfg['username'],
        passwd=out_cfg['passwd'],
        id=cfg['id'],
        topic=out_cfg['topic']  # if in db mode, can be omitted
    )

    node_cfg = {
        'id': cfg['id'],
        'name': cfg['name'],
        'model_dir': cfg['model_dir'],  # could be file path or url or model name
        'max_task': cfg['max_task'], # only effective in batch mode
        'mode': cfg['mode'],
        'task': task,
        'alg': alg,
        'out': submitter
    }

    node = AlgNode(max_task=10, cfg=node_cfg, task=task)
    node_web_api = AlgNodeWeb(cfg['web_port'], node)

    node.start()
    node_web_api.run()

    print('start node')
    while True:
        time.sleep(5)
        # node.stop()
        # print('stop node, exit')
        # break

if __name__ == '__main__':
    main()

