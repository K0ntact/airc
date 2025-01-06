import os

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class ATMADataset(Dataset):
    def __init__(self, vid_folder_path, label_path, seq_len=30):
        self.vid_folder_path = vid_folder_path
        self.label_path = label_path
        self.label_idx = {'normal': 0, 'anomaly': 1}
        self.seq_len = seq_len

        # Sliding window parameters
        self.buffer_size = 16
        self.frame_sample_per_second = 8

        self.video_paths = self._get_input_path()
        self.anomaly_span = self._get_anomaly_spans()
        self.video_tensor_idx_mapping = self._create_video_tensor_idx_mapping()
        self.video_tensor_sequence_mapping = self._get_tensor_sequence()

    def __len__(self):
        return len(self.video_tensor_sequence_mapping)

    def __getitem__(self, idx):
        vid_path, tensor_seq = self.video_tensor_sequence_mapping[idx]
        seq_frame_tensors = []
        seq_labels = []
        for tensor_idx in tensor_seq:
            frame_buffer, label = self._load_frame_tensor(vid_path, tensor_idx)
            seq_frame_tensors.append(frame_buffer)
            seq_labels.append(label)

        seq_frame_tensors = torch.stack(seq_frame_tensors, dim=0)  # (seq_len, T, C, H, W)

        # Get label for each tensor in the sequence
        # TODO: pad seq_frame_tensors and seq_labels to self.seq_len
        # seq_labels = torch.stack(seq_labels, dim=0)  # (seq_len, 2)

        # Get only the last label
        seq_labels = seq_labels[-1]
        return seq_frame_tensors, seq_labels

    def _get_input_path(self) -> list[str]:
        """
        Read the folder and returns list of sorted video paths
        """
        video_paths = []
        for root, dirs, files in os.walk(self.vid_folder_path):
            for file in files:
                video_paths.append(os.path.join(root, file))
        return sorted(video_paths)

    def _get_anomaly_spans(self):
        """
        Extract anomaly spans for each video from label file.
        Anomaly spans are rescaled since input videos were augmented to have 30fps.

        :return: List of rescaled anomaly spans for each video
        """
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        all_anomaly_spans = []
        for path in self.video_paths:
            vid = os.path.basename(path)
            og_vid = vid.split("-")[0]
            og_vid = str(og_vid) + ".mp4"

            for line in lines:
                # og_name, total_frames, frame_rate, start_span_1, end_span_1, start_span_2, end_span_2,....
                og_name = line.strip().split()[0]
                old_fps = int(line.strip().split()[2])
                if og_vid == og_name:
                    # Since aug videos are modified to have 30 fps, we need to scale the anomaly spans
                    # Approach: divide anomaly span with old fps and multiply with 30 fps
                    anomaly_spans = line.strip().split()[3:]
                    rescaled_span = []
                    for i in range(0, len(anomaly_spans), 2):
                        start = int(anomaly_spans[i])
                        end = int(anomaly_spans[i + 1])
                        new_start = int(start * 30 // old_fps)
                        new_end = int(end * 30 // old_fps)
                        rescaled_span.append((new_start, new_end))
                    all_anomaly_spans.append(rescaled_span)
                    break
        return all_anomaly_spans

    def _create_video_tensor_idx_mapping(self):
        """
        Create a mapping of video path and buffer tensor index that can be extract
        """
        mapping = []
        for vid_path in self.video_paths:
            cap = cv2.VideoCapture(vid_path)
            vid_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            vid_duration = vid_total_frames / vid_fps
            target_frames = np.linspace(0,
                                        vid_total_frames - 1,
                                        num=int(vid_duration * self.frame_sample_per_second),
                                        dtype=np.int32)

            n_tensors = len(target_frames) // self.frame_sample_per_second - 1  # Maximum number of frame buffer tensors we can extract
            for tensor_idx in range(n_tensors):
                mapping.append((vid_path, tensor_idx))
        return mapping

    def _load_frame_tensor(
            self,
            video_path: str,
            tensor_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load specific frame tensor from video by skipping to correct position.
        Extract label for the tensor based on anomaly spans.

        :param video_path: Path to video file
        :param tensor_idx: Which tensor to extract (0-based index)
        :return: Tuple of frame tensor and label
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            # Get target frame numbers for this tensor
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            target_frames = np.linspace(
                0,
                total_frames - 1,
                num=int(duration * self.frame_sample_per_second),
                dtype=np.int32
            )

            frames_per_tensor = self.buffer_size
            target_frames_to_skip = tensor_idx * self.frame_sample_per_second

            start_pos = target_frames_to_skip
            end_pos = start_pos + frames_per_tensor
            if end_pos > len(target_frames):
                raise IndexError(f"Tensor index {tensor_idx} out of bounds")

            # Determine label for this tensor
            # label: array of 2 elements, 0th element is normal, 1st element is anomaly
            label = [0, 0]
            input_index = self.video_paths.index(video_path)
            anomaly_spans = self.anomaly_span[input_index]
            start_frame = target_frames[start_pos]
            end_frame = target_frames[end_pos - 1]
            for span in anomaly_spans:
                if span[0] == -1:
                    label[0] = 1
                    break
                if span[0] < start_frame < span[1] or span[0] < end_frame < span[1]:
                    label[1] = 1
                    break
                else:
                    label[0] = 1

            if label[0] == 0 and label[1] == 0:
                raise RuntimeError(f"Label not found for video {video_path} at tensor {tensor_idx}")

            label = torch.tensor(label, dtype=torch.float32)

            # Get the frames we need for this tensor
            tensor_target_frames = target_frames[start_pos:end_pos]
            frames = []
            current_frame = 0
            for target_idx in tensor_target_frames:
                while current_frame < target_idx:
                    cap.read()
                    current_frame += 1

                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to read frame {current_frame}")
                frames.append(frame)
                current_frame += 1

            if len(frames) != self.buffer_size:
                raise RuntimeError(f"Expected {self.buffer_size} frames, got {len(frames)}")

            frame_tensor = np.stack(frames, axis=0)  # (T, H, W, C)
            frame_tensor = torch.from_numpy(frame_tensor).float()
            frame_tensor = frame_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

            return frame_tensor, label

        finally:
            cap.release()

    def _get_tensor_sequence(self):
        """
        Extract sequence of frame tensor with length seq_len for each video
        """
        # (1.mp4, 1), (1.mp4, 2), (1.mp4, 3),...
        # seq_len = 5
        # Output: (1.mp4, [1,2,3,4,5]), (1.mp4, [2,3,4,5,6]), (1.mp4, [3,4,5,6,7]),...

        mapping = []
        seq = []
        previous_vid_path = ""
        previous_tensor_idx = -1
        for tup in self.video_tensor_idx_mapping:
            vid_path, tensor_idx = tup

            if (previous_tensor_idx + 1) != tensor_idx:    # new video
                if len(seq) != self.seq_len and previous_tensor_idx < self.seq_len:    # end of a vid with total tensors < seq len
                    mapping.append((previous_vid_path, seq))
                seq = []

            previous_vid_path = vid_path
            previous_tensor_idx = tensor_idx
            seq.append(tensor_idx)
            if len(seq) == self.seq_len:    # end of a vid with total tensors >= seq len
                mapping.append((vid_path, seq))
                seq = seq[1:]

        return mapping
