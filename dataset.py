import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class ATMADataset(Dataset):
    def __init__(self, vid_folder_path, label_path):
        self.label_idx = {'normal': 0, 'anomaly': 1}
        self.input_paths = self.get_input_path(vid_folder_path)
        self.labels = self.get_labels(label_path)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        return self.vid_to_tensor(self.input_paths[idx]), self.labels[idx]

    def get_input_path(self, folder_path):
        """
        Read the folder and returns list of video paths
        """
        video_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                video_paths.append(os.path.join(root, file))
        return video_paths

    def get_labels(self, label_path):
        """
        Read label file and return a list of labels, where label order match with the corresponding video order
        """
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = []
        for path in self.input_paths:
            vid = os.path.basename(path)

            for line in lines:
                vid_name, tag = line.strip().split()
                if vid == vid_name:
                    labels.append(self.label_idx[tag])
                    break
        return labels

    def vid_to_tensor(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        frame_tensor = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor.append(frame)
        cap.release()

        frame_tensor = torch.Tensor(np.array(frame_tensor, dtype=np.float32))
        return frame_tensor/255.0
