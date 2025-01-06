from transformers import TimesformerModel, AutoImageProcessor
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch import Tensor


class TimesformerGRU(nn.Module):
    def __init__(
            self,
            pretrained_tsf: str,
            gru_hidden_size: int,
            gru_layers: int,
            num_classes: int,
            dropout: float = 0.1,
    ) -> None:
        """
        Initialize TimesformerGRU model.

        Args:
            pretrained_tsf (str): Name or path of pretrained Timesformer model
            gru_hidden_size (int): Hidden size of GRU layers
            gru_layers (int): Number of GRU layers
            num_classes (int): Number of output classes
            dropout (float): Dropout rate for regularization
        """
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(pretrained_tsf)
        self.timesformer = TimesformerModel.from_pretrained(pretrained_tsf)

        # No Timesformer training for now
        for param in self.timesformer.parameters():
            param.requires_grad = False

        self.gru = nn.GRU(
            input_size=768,  # Timesformer's hidden size
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    @torch.no_grad()
    def extract_features(self, window: Tensor) -> Tensor:
        """
        Extract features from a window of frames using Timesformer.

        Args:
            window (Tensor): Input window of shape (window_length, channels, height, width)

        Returns:
            Tensor: CLS tokens for the window
        """
        x_list = [frame for frame in window]
        inputs = self.processor(images=x_list, return_tensors="pt")
        inputs = {k: v.to(window.device) for k, v in inputs.items()}

        outputs = self.timesformer(**inputs)
        return outputs[0][:, 0, :]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, gru_input_length, window_length, channels, height, width)

        Returns:
            Tensor: Classification logits for each window in the input tensor (batch, gru_input_length, num_classes)
        """
        batch_size, seq_length = x.shape[:2]

        cls_tokens = []
        for batch_idx in range(batch_size):
            sequence_tokens = []
            for seq_idx in range(seq_length):
                cls_token = self.extract_features(x[batch_idx, seq_idx])
                sequence_tokens.append(cls_token)
            cls_tokens.append(torch.cat(sequence_tokens, dim=0))

        cls_token_batch = torch.stack([
            tokens.view(seq_length, -1) for tokens in cls_tokens
        ], dim=0)  # Shape: (batch_size, seq_length, hidden_size)

        output, final_hidden = self.gru(cls_token_batch)    # Output Shape: (batch_size, seq_length, gru_hidden_size)

        # Return final hidden output
        predictions = []
        for batch_idx in range(batch_size):
            prediction = self.classifier(final_hidden[batch_idx])
            predictions.append(prediction)
        predictions = torch.stack(predictions, dim=0).squeeze(1)   # Shape: (batch_size, num_classes)

        # Return all output of GRU
        # predictions = []
        # for batch in range(batch_size):
        #     predict_batch = []
        #     for seq in range(seq_length):
        #         prediction = self.classifier(output[batch, seq])
        #         predict_batch.append(prediction)
        #     predictions.append(torch.stack(predict_batch, dim=0))
        # predictions = torch.stack(predictions, dim=0)   # Shape: (batch_size, seq_length, num_classes)

        predictions = softmax(predictions, dim=-1)
        return predictions
