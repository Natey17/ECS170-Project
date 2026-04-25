'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-3
    loss_history = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # ===== Model architecture: 784 -> 256 -> 128 -> 10 =====
        self.fc_layer_1 = nn.Linear(784, 256)
        self.fc_layer_2 = nn.Linear(256, 128)
        self.fc_layer_3 = nn.Linear(128, 10)

        self.activation_func_1 = nn.ReLU()
        self.activation_func_2 = nn.ReLU()

    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        y_pred = self.fc_layer_3(h)
        return y_pred

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X)) / 255.0
        y_tensor = torch.LongTensor(np.array(y))

        self.loss_history = []

        for epoch in range(self.max_epoch):
            self.train()
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            self.loss_history.append(train_loss.item())

            if epoch % 10 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_tensor,
                    'pred_y': y_pred.argmax(dim=1)
                }
                print('--------------------------------------------------')
                print('Epoch:', epoch,
                      'Accuracy:', accuracy_evaluator.evaluate(),
                      'Loss:', round(train_loss.item(), 4))

    def test(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X)) / 255.0
            y_pred = self.forward(X_tensor)
        return y_pred.argmax(dim=1)

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'loss_history': self.loss_history
        }