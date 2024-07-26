import os
import glob
import time
import cv2
import numpy as np
import torch
from ucs_alg_node import AlgNode, Alg, AlgSubmitter, AlgResult, AlgNodeWeb, AlgTask, utils
from models.model import Fcn  # 确保模型的导入

weights_path = "weights/model_12.pth"  # 模型权重路径
video_resize = 0.25  # 全局参数

task_id = 'task112'
node_id = 'node001'
alg_name = 'my_alg'
out_topic = 'ucs/alg/result'



class MyAlg(Alg):
    def __init__(self, mode, sources, model, id):
        super().__init__(mode, model)
        self.name = 'my_alg'
        self.sources = sources
        self.id = id

        # 初始化模型
        self.model = Fcn()
        self.model = self.model.to(self.model.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.model.device))
        self.model.eval()

    def infer_batch(self, task):
        vid_file = task.sources[0]
        cap = cv2.VideoCapture(vid_file)
        if not cap.isOpened():
            print(f"无法打开视频文件: {vid_file}")
            return []
        ret, img = cap.read()
        if ret == False:
            print("读取视频失败")
            cap.release()
            cv2.destroyAllWindows()
            return []

        img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)
        img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)
        prev_img_grey = img_grey
        labels = []

        while True:
            ret, img = cap.read()
            if not ret:
                break

            img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)
            img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_img_grey, img_grey, None, 0.5, 5, 15, 3, 5, 1.1,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            flow = flow[np.newaxis, :]
            flow = flow.transpose(0, 3, 1, 2)
            flow = torch.from_numpy(flow).float().to(self.model.device)

            predict = self.model(flow)
            predict = torch.argmax(predict, 1).cpu().numpy()[0]

            labels.append(predict)
            prev_img_grey = img_grey

        cap.release()
        cv2.destroyAllWindows()

        ts = time.time_ns()
        result = AlgResult(task.id, ts, labels, "inference result")
        return result


def main():
    # vid_root = 'C:/ucs_alg_node_test-master/ucs_alg_node_test/Motion_Emotion_Dataset'  # 视频文件夹路径
    # vid_files = glob.glob(os.path.join(vid_root, '*.mp4'))

    # 配置
    cfg = {
        'id': node_id,
        'name': alg_name,
        'mode': 'batch',
        'max_task': 10,
        'model_dir': os.path.join(utils.get_cwd(), 'models'),
        'alg_id': 'alg_id123',
        'web_port': 9996
    }
    task = AlgTask(id=task_id,
                   ts=utils.current_time_milli(),
                   sources=['C:/ucs_alg_node_test-master/ucs_alg_node_test/Motion_Emotion_Dataset/001.mp4']
                   # ['rtsp://localhost:9111/123',
                   #          'mqx://localhost:8011/1123']
                   )
    alg_cfg = {
        'alg_id': 'alg_id123',
        'model': 'model_12.pth'
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
        topic=out_cfg['topic']
    )

    # 创建和启动节点


    node_cfg = {

        'id': cfg['id'],
        'name': cfg['name'],
        'model_dir': cfg['model_dir'],  # could be file path or url or model name
        'max_task': cfg['max_task'],  # only effective in batch mode
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
    result = alg.infer_batch(task)
    print(f"Task {task.id} labels: {result.labels}")
    while True:
        time.sleep(5)

        # node.stop()
        # print('stop node, exit')
        # break

    # 处理每一个视频文件
    # for vid_file in vid_files:
    #     tic = time.time()
    #     print('Processing', vid_file)
    #
    #     # 创建任务并设置 sources
    #
    #
    #     # 更新算法的 sources
    #     alg.sources = ['C:/ucs_alg_node_test-master/ucs_alg_node_test/Motion_Emotion_Dataset']
    #
        # result = alg.infer_batch(task)
        # print(result)
        # toc = time.time()
        # print('Time cost:', toc - tic)
        #
        # if result:
        #     print(f"Result: {result}")
    #         submitter.submit(result)  # 将结果提交到 MQTT 主题


if __name__ == '__main__':
    main()