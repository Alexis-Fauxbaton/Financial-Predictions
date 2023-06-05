import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
import numpy as np
import time


def get_accuracy(output, target):
    y_true = target.detach().numpy()

    y_prob = output.detach().numpy()

    y_pred = np.argmax(y_prob, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    accuracy = metrics.accuracy_score(y_true, y_pred)

    return accuracy


def get_confusion_matrix(output, target):
    y_true = target.detach().numpy()
    y_prob = output.detach().numpy()
    y_pred = np.argmax(y_prob, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


class TSDataset(Dataset):

    def __init__(self, data: pd.DataFrame, seq_length=15) -> None:
        super().__init__()

        self.data = data.copy()
        self.unix = data["Unix"]
        self.target = data['Target']
        self.output_size = self.target.unique().size
        self.data.drop(["Unix", "Target"], axis=1, inplace=True)
        self.seq_length = seq_length
        self.size = len(self.data.columns)

    def __len__(self):
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, index):
        seq = self.data.loc[index:index + self.seq_length].values
        target = int(self.target.loc[index + self.seq_length].tolist())
        target_tensor = torch.zeros(self.output_size)
        if self.output_size == 3:
            target_tensor[target + 1] = 1
        else:
            target_tensor[target] = 1
        return torch.Tensor(seq), target_tensor


def train_lstm(model, train_dataset, val_dataset, epochs=30, lr=0.01, batch_size=128, num_layers=3, hidden_size=100,
               device='CPU', train_sampler=None, class_weights=None):

    if train_sampler is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss(
    ) if class_weights is None else nn.CrossEntropyLoss(weight=class_weights)

    losses = []

    val_losses = []

    accuracies = []

    val_accuracies = []

    best_confusion_matrix = None
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(epochs):
        model = model.train()

        start_time = time.process_time()

        running_loss = 0

        running_accuracy = 0

        batch_idx = 0

        data = None

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            data = data.squeeze(-1)

            hidden = (torch.zeros(num_layers, data.shape[0], hidden_size).to(device), torch.zeros(
                num_layers, data.shape[0], hidden_size).to(device))  # Hidden state and cell state
            # data.shape[0] : batch_size

            # print(data.shape, hidden[0].shape)

            output, _ = model(data, hidden)

            # print("Output 1", output.shape, output)

            # output = output.view(-1)

            output = torch.squeeze(output[:, -1:, :], 1)

            # print("Output 2", output.shape, output)

            loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            accuracy = get_accuracy(output.cpu(), target.cpu())

            running_accuracy += accuracy * len(data)

            print("Epoch: {}/{} -- [{}/{} ({:.1f}%)]\tLoss: {}".format(
                epoch + 1, epochs, (batch_idx + 1) *
                len(data), len(train_loader.dataset),
                100 * (batch_idx + 1) / len(train_loader), running_loss / (batch_idx + 1)), end='\r')

        end_time = time.process_time()

        val_running_loss = 0

        val_running_accuracy = 0

        with torch.no_grad():

            model = model.eval()

            val_running_confusion_matrix = np.zeros((3, 3))
            val_data_buffer = None
            val_target_buffer = None

            for val_batch_idx, (val_data, val_target) in enumerate(val_loader):
                val_data, val_target = val_data.to(
                    device), val_target.to(device)

                val_data = val_data.squeeze(-1)

                val_hidden = (torch.zeros(num_layers, val_data.shape[0], hidden_size).to(device), torch.zeros(
                    num_layers, val_data.shape[0], hidden_size).to(device))  # Hidden state and cell state

                # print(val_data.shape, val_hidden[0].shape)

                val_output, _ = model(val_data, val_hidden)

                val_output = torch.squeeze(val_output[:, -1:, :], 1)

                val_loss = criterion(val_output, val_target)

                val_running_loss += val_loss

                val_running_confusion_matrix += get_confusion_matrix(
                    val_output.cpu(), val_target.cpu())

                accuracy = get_accuracy(val_output.cpu(), val_target.cpu())

                val_running_accuracy += accuracy * len(val_data)

        val_acc = np.sum(np.diag(val_running_confusion_matrix)
                         ) / np.sum(val_running_confusion_matrix)
        if val_acc > best_acc:
            best_acc = val_acc
            best_confusion_matrix = val_running_confusion_matrix

        print("Epoch: {}/{} -- [{}/{} ({:.1f}%)]\tLoss: {}\tAccuracy: {:.3f}\tTime taken: {}".format(
            epoch + 1, epochs, (batch_idx + 1) *
            len(data), len(train_loader.dataset),
            100 * (batch_idx + 1) /
            len(train_loader), running_loss / (batch_idx + 1),
            running_accuracy / len(train_loader.dataset), end_time - start_time), end='\t')

        print("Validation Loss: {} || Validation Accuracy: {:.3f}".format(
            val_running_loss / (val_batch_idx + 1), val_running_accuracy / len(val_loader.dataset)))

        losses.append(running_loss / (batch_idx + 1))
        val_losses.append(val_running_loss / (val_batch_idx + 1))

        accuracies.append(running_accuracy / len(train_loader.dataset))
        val_accuracies.append(val_running_accuracy / len(val_loader.dataset))

    print(
        f"Best accuracy : {best_acc} || Best confusion matrix : \n", best_confusion_matrix)

    print("Last confusion matrix : \n", val_running_confusion_matrix)

    return losses, accuracies, val_losses, val_accuracies, best_confusion_matrix, best_acc


def eval_lstm(model, dataset, batch_size=128, num_layers=3, hidden_size=100, device='cpu'):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    loss = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    confusion_matrix = np.zeros((3, 3))

    accuracy = 0

    outputs = []
    targets = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        data = data.squeeze(-1)

        hidden = (torch.zeros(num_layers, data.shape[0], hidden_size).to(device),
                  torch.zeros(num_layers, data.shape[0], hidden_size).to(device))  # Hidden state and cell state

        output, _ = model(data, hidden)

        output = torch.squeeze(output[:, -1:, :], 1)
        
        outputs.append(output)
        targets.append(target)

        loss += criterion(output, target) * len(data)

        confusion_matrix += get_confusion_matrix(output.cpu(), target.cpu())

        accuracy += get_accuracy(output.cpu(), target.cpu()) * len(data)

    loss /= len(dataloader.dataset)

    accuracy /= len(dataloader.dataset)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    print(f"Accuracy : {accuracy} || Loss : {loss}")

    print(f"Confusion matrix : \n{confusion_matrix}")
    
    return outputs, targets