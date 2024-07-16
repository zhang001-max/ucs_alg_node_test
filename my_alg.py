import os
import shutil
from pathlib import Path
import cv2
import torch
import time
import numpy as np
from tqdm import tqdm

from src.ucs_alg_node import Alg, Fcn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Fcn()  # 定义模型
weights_path = "test/weights/model_12.pth"  # 权重
weights = torch.load(weights_path,map_location='cpu')
model.load_state_dict(weights)
model.to(device)


def download_video(video_url, save_path):
    """Downloads videos from the given video paths."""
    # Download the video
    # copy the downloaded video to the save_path

    shutil.copy(video_url, save_path)
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    return video_path, cur_time


def get_frame_count(video_path):
    """Returns the total number of frames in the video."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return -1
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


class MyAlg(Alg):
    def rename_batch(self, video_path, save_folder):
        video_path, cur_time = download_video(video_path, 'video.mp4')
        frame_count = get_frame_count(video_path)
        video_path = Path(video_path)
        new_filepath = None
        if frame_count != -1:
            new_filename = f"{video_path.stem}_{cur_time}.{video_path.suffix}"
            new_filepath = os.path.join(save_folder, new_filename)
            shutil.copy(video_path, new_filepath)
            print(f"Copy '{video_path}' to '{new_filepath}'")
        else:
            print(f"Failed to read '{video_path.stem}'")

        # TODO: 删除临时文件
        os.remove('video.mp4')
        return frame_count, new_filepath

    def infer_batch(self, data, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        frame_count, video_path = self.rename_batch(data, save_folder)

        cap = cv2.VideoCapture(video_path)  # 读取视频
        ret, img = cap.read()  # 取出
        if ret == False:
            print("读取视频失败")
            cap.release()
            cv2.destroyAllWindows()
            return []

        img_model = cv2.resize(img, (512, 512))  # 输入模型的第一帧图片，缩放
        img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)  # 二值
        prev_img_grey = img_grey  # 第一帧视频
        frame = 1  # 第二帧视频

        labels = []
        pbar = tqdm(total=frame_count)
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
            flow = torch.from_numpy(flow).float().to(device)

            predict = model(flow)  # 预测输出
            predict = torch.argmax(predict, 1).cpu().numpy()[0]

            labels.append(predict)

            frame += 1  #
            prev_img_grey = img_grey
            pbar.update(1)
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()
        return labels


if __name__ == "__main__":
    data = r'D:\gugol\Motion_Emotion_Dataset'
    save_folder = 'test/videos'
    alg = MyAlg()

    files = os.listdir(data)
    print(f"Files in directory: {files}")
    videos = [os.path.join(data, file) for file in files if file.endswith('.mp4')]  # Filter for .mp4 files

    for video_path in videos:
        video = alg.infer_batch(video_path, save_folder)
