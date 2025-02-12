import torch
import torch.nn as nn
from tensorflow.python.feature_column.utils import sequence_length_from_sparse_tensor
from torch import Tensor
from transformers import TimesformerModel


class CustomTimesformer(nn.Module):
    def __init__(self,
                 pretrained_tsf: str,
                 num_classes: int):
        super().__init__()
        self.timesformer = TimesformerModel.from_pretrained(pretrained_tsf)
        self.classifier = nn.Linear(self.timesformer.config.hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape (batch, window_length, channels, height, width)
        :return: Classification logits
        """

        with torch.no_grad():
            batch_cls_tokens = self.timesformer(x)[0][:, 0, :]
        return self.classifier(batch_cls_tokens)
