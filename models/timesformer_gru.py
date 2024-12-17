from transformers import TimesformerModel, AutoImageProcessor
import torch
import torch.nn as nn
import numpy as np


class TimesformerGRU(nn.Module):
    def __init__(self, pretrained_tsf: str, gru_hidden_size: int, gru_layers: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoImageProcessor.from_pretrained(pretrained_tsf)
        self.timesformer = TimesformerModel.from_pretrained(pretrained_tsf)
        self.gru = nn.GRU(input_size=768, hidden_size=gru_hidden_size, num_layers=gru_layers, batch_first=True)
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x, *args, **kwargs):
        inputs = self.processor(images=x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.timesformer(**inputs)
            cls_reps = outputs[0][:, 0, :]
        _, hidden = self.gru(cls_reps.unsqueeze(0))
        return self.classifier(hidden[-1])
