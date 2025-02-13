import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
# from torchinfo import summary
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
import wandb

from dataset import TimesformerData
from models.timesformer import CustomTimesformer

model = CustomTimesformer(pretrained_tsf="facebook/timesformer-base-finetuned-k600",
                          num_classes=2)


class TrainingLoop:
    def __init__(self, model, dataloader, optimizer, scheduler, num_epochs, device):
        self.model = model
        self.train_dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device

    def _grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_grad = p.grad
            if param_grad is not None:
                total_norm = (param_grad ** 2).sum().item()
        total_norm = np.sqrt(total_norm)
        return total_norm

    def train(self):
        self.model.to(self.device)
        self.model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            epoch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for step, (inputs, labels) in enumerate(epoch_iterator):
                # for step, (inputs, labels) in enumerate(self.train_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)

                # print("="*50)
                # print(f"Step: {step}")
                # print(f"Prediction: {outputs}")
                # print(f"Label: {labels}")
                # print(f"Loss: {loss.item()}")
                # print("="*50)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                grad_norm = self._grad_norm()
                wandb.log({"batch_loss": loss.item(), "grad_norm": grad_norm, "epoch": epoch})

                epoch_iterator.set_postfix(loss=loss.item())

        # Save model
        torch.save(self.model.state_dict(), "timesformer.pth")


if __name__ == "__main__":
    batch_size = 8
    lr = 1e-5
    warm_up_steps = 100
    num_epochs = 3

    dataset = TimesformerData(vid_folder_path="./datasets/ATMA-V/videos/train/aug",
                              label_path="./datasets/ATMA-V/labels/labels.txt")

    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                num_training_steps=len(train_dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    wandb.init(project="airc")
    wandb.config.update({
        "batch_size": batch_size,
        "learning_rate": lr,
        "warm_up_steps": warm_up_steps,
        "num_epochs": num_epochs})
    training_loop = TrainingLoop(model=model,
                                 dataloader=train_dataloader,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 num_epochs=num_epochs,
                                 device=device)
    training_loop.train()

# Avg model run time: 0.08
# Avg 1 step time: 1.77s
# Bottleneck: Data load time: 2-10s