from transformers import TimesformerModel
import torch
import torch.nn as nn
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

        self.timesformer = TimesformerModel.from_pretrained(pretrained_tsf)

        # No Timesformer training for now
        for param in self.timesformer.parameters():
            param.requires_grad = False

        self.input_norm = nn.LayerNorm(768)

        self.gru = nn.GRU(
            input_size=768,  # Timesformer's hidden size
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.output_norm = nn.LayerNorm(gru_hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, gru_input_length, window_length, channels, height, width)

        Returns:
            Tensor: Classification logits
        """
        batch_size, seq_length = x.shape[:2]

        cls_tokens = []
        for batch_idx in range(batch_size):
            with torch.no_grad():
                outputs = self.timesformer(x[batch_idx])
            seq_cls_tokens = outputs[0][:, 0, :]  # Shape: (seq_length, hidden_size)
            cls_tokens.append(seq_cls_tokens)

        cls_token_batch = torch.stack([
            tokens.view(seq_length, -1) for tokens in cls_tokens
        ], dim=0)  # Shape: (batch_size, seq_length, hidden_size)
        cls_token_batch = self.input_norm(cls_token_batch)

        torch.nn.utils.clip_grad_norm_(self.gru.parameters(), max_norm=1.0)

        output, final_hidden = self.gru(cls_token_batch)
        # Output Shape: (batch_size, seq_length, gru_hidden_size)
        # Final Hidden Shape: (num_layers, batch_size, gru_hidden_size)

        # Return final hidden output of the last layer
        norm_last_layer_hidden = self.output_norm(final_hidden[-1]) # (batch_size, gru_hidden_size)
        logits = self.classifier(norm_last_layer_hidden)
        
        # Return all output of GRU
        # predictions = []
        # for batch in range(batch_size):
        #     predict_batch = []
        #     for seq in range(seq_length):
        #         prediction = self.classifier(output[batch, seq])
        #         predict_batch.append(prediction)
        #     predictions.append(torch.stack(predict_batch, dim=0))
        # predictions = torch.stack(predictions, dim=0)   # Shape: (batch_size, seq_length, num_classes)

        return logits
