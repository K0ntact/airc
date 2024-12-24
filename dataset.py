import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class ATMADataset(Dataset):
    def __init__(self, vid_folder_path, label_path):
        self.label_idx = {'normal': 0, 'anomaly': 1}

        self.buffer_size = 16
        self.small_vid_length_s = 1
        self.frames_per_small_vid = 8

        self.anomaly_span = []
        self.input_tensors = None
        self.input_paths = self.get_input_path(vid_folder_path)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        vid_tensor = self.vid_to_tensor(self.input_paths[idx])
        frame_indices = np.linspace(0, vid_tensor.shape[0] - 1, 8).astype(int)
        vid_tensor = vid_tensor[frame_indices]
        return vid_tensor, self.labels[idx]

    def get_input_path(self, folder_path):
        """
        Read the folder and returns list of video paths
        """
        video_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                video_paths.append(os.path.join(root, file))
        return sorted(video_paths)

    def get_labels(self, label_path):
        """
        Read label file and return a list of labels, where label order match with the corresponding video order
        """
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = []
        for path in self.input_paths:
            vid = os.path.basename(path)
            og_vid = vid.split("-")[0]

            for line in lines:
                # og_name, total_frames, frame_rate, start_span_1, end_span_1, start_span_2, end_span_2,....
                og_name = line.strip().split()[0]
                anomaly_spans = line.strip().split()[3:]
                anomaly_spans = [(int(anomaly_spans[i]), int(anomaly_spans[i + 1])) for i in range(0, len(anomaly_spans), 2)]
                if og_vid == og_name:
                    self.anomaly_span.append(anomaly_spans)
                    break
        return labels

    def vid_to_tensor(self, vid_path):
        """
        Read the video and returns a tensor of shape (frames, 3, 224, 224)
        """
        cap = cv2.VideoCapture(vid_path)
        frame_tensor = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            frame_tensor.append(frame)
        cap.release()

        frame_tensor = torch.Tensor(np.array(frame_tensor))
        return frame_tensor/255.0
