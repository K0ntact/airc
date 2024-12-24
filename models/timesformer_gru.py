from transformers import TimesformerModel, AutoImageProcessor
import torch
import torch.nn as nn
import numpy as np


class TimesformerGRU(nn.Module):
    def __init__(self, pretrained_tsf: str, gru_hidden_size: int, gru_layers: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoImageProcessor.from_pretrained(pretrained_tsf)
        self.timesformer = TimesformerModel.from_pretrained(pretrained_tsf)
        self.gru = nn.GRU(input_size=768, hidden_size=gru_hidden_size, num_layers=gru_layers, batch_first=False)
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    def __batched_tensor_to_list(self, batched_tensor: torch.Tensor):
        """
        Convert batched tensor of videos to list of videos, where each video is a list of frames, each frame is a tensor
        """
        list_tensor = []
        for vid_tensor in batched_tensor.unbind(0):
            list_tensor.append(vid_tensor.unbind(0))
        return list_tensor

    def forward(self, x, *args, **kwargs):
        # x shape: batch, frames, channels, height, width
        x_device = x.device
        x_list = self.__batched_tensor_to_list(x)

        # Processor only takes in list of tensors
        inputs = self.processor(images=x_list, return_tensors="pt")
        inputs.to(x_device)

        with torch.no_grad():
            outputs = self.timesformer(**inputs)
            cls_reps = outputs[0][:, 0, :]  # batch, hidden_size
        _, hidden = self.gru(cls_reps.unsqueeze(0))
        return self.classifier(hidden[-1])
