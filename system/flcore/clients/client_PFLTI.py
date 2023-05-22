import numpy as np
import torch
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader


class clientPFLTI(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate
        # self.mom_model = copy.deepcopy(args.model)
        self.optimizer_global = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_mom = PerAvgOptimizer(self.mom_model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

    # def get_optimizer(self, model):
    #     param = model.parameters()
    #     optimizer = PerAvgOptimizer(model.parameters(), lr=self.learning_rate0)
    #     return optimizer

    def train(self):
        trainloader = self.load_train_data(self.batch_size*2)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        self.dropout_eval(self.model)
        self.mom_model.train()
        self.dropout_eval(self.mom_model)

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            for X, Y in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))\

                # step 1: Adaptation for global model and momentum model

                # Data preprocessing
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Update global model params
                self.optimizer_global.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer_global.step()

                # Apply dropout
                # After training then dropout
                self.model.train()
                # self.mom_model.train()

                # Update momentum model params
                self.optimizer_mom.zero_grad()
                output = self.mom_model(x)
                loss_mom = self.loss(output, y)
                loss_mom.backward()
                self.optimizer_mom.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_global = self.loss(output, y)
                mom_output = self.mom_model(x)
                kl_loss = self.KL_div(output, mom_output, temp=4)
                loss = (1 - 0.5) * loss_global + 0.5 * kl_loss
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer_global.step(beta=self.beta)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def dropout_eval(self, m):
        def _is_leaf(model):
            return len(list(model.children())) == 0

        if hasattr(m, 'dropout'):
            m.dropout.eval()

        for child in m.children():
            if not _is_leaf(child):
                self.dropout_eval(child)

    def KL_div(self, student_output, teacher_output, temp):
        p_s = F.log_softmax(student_output / temp, dim=1)
        p_t = F.softmax(teacher_output / temp, dim=1)
        return F.kl_div(p_s, p_t, size_average=False) * (temp ** 2)

    def train_one_step(self):
        trainloader = self.load_train_data_one_step(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()

        (x, y) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()

        # self.model.cpu()


    def load_train_data_one_step(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)