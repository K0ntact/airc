from torch import Tensor
import torch.nn as nn
from torch.nn import LSTM, Linear, Dropout, ReLU, BatchNorm1d, LayerNorm
from torchvision.models import resnet50


class DenseReLU(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(DenseReLU, self).__init__()
        self.linear = Linear(input_size, output_size)
        self.bn = BatchNorm1d(output_size)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResNetAvgPool(nn.Module):
    def __init__(self):
        """
        Return only the extracted features from avgpool layer of ResNet model
        """
        super(ResNetAvgPool, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.layers = nn.Sequential(*list(self.resnet.children())[:-1])
        # for layer in self.layers[:20]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        """
        Input shape: (seq_len, 3, 224, 224)
        Output shape: (seq_len, 2048)
        """
        return self.layers(x)


class ResNetBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBiLSTM, self).__init__()
        self.resnet = ResNetAvgPool()
        self.lstm = LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = Dropout(0.25)
        self.layernorm = LayerNorm(1024)
        self.bn1 = BatchNorm1d(1024)
        self.bn2 = BatchNorm1d(512)
        self.bn3 = BatchNorm1d(256)
        self.dense1 = DenseReLU(1024, 512, 0.25)
        self.dense2 = DenseReLU(512, 256, 0.25)
        self.dense3 = DenseReLU(256, 128, 0.25)
        self.classifier = Linear(128, num_classes)

    def forward(self, x: Tensor):
        """
        Input shape: (batch, seq_len, 3, 224, 224)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        x_resnet_reshaped = x.view(batch_size * seq_len, *x.size()[2:])  # (batch * seq_len, 3, 224, 224)
        x = self.resnet(x_resnet_reshaped)  # (batch * seq_len, 2048, 1, 1)
        x_lstm_reshaped = x.view(batch_size, seq_len, -1)  # (batch, seq_len, 2048)
        _, (h, _) = self.lstm(x_lstm_reshaped)  # (2, batch, 512)
        h = h.permute(1, 0, 2).reshape(batch_size, -1)  # (batch, 1024)
        x = self.layernorm(h)
        x = self.dropout(x)

        x = self.bn1(x)  # (batch, 1024)
        x = self.dense1(x)  # (batch, 512)

        x = self.bn2(x)  # (batch, 512)
        x = self.dense2(x)  # (batch, 256)

        x = self.bn3(x)  # (batch, 256)
        x = self.dense3(x)  # (batch, 128)
        x = self.classifier(x)  # (batch, num_classes)
        return x
