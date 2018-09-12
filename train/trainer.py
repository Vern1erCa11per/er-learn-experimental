import csv

import torch
from tqdm import tqdm

# FIXME replace print with logger
class Trainer(object):
    def __init__(self, model:torch.nn.Module, optimizer, optimizer_args, loss_function, save_model_path,
                 save_loss_log, save_best_only=True, patient=None, use_gpu=None):
        self.model = model
        if use_gpu and torch.cuda.is_available():
            self.use_gpu=True
            self.model.cuda(0)
            self.loss_function = loss_function.cuda(0)

        else:
            self.loss_function = loss_function
            self.use_gpu = False

        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)

        self.accumulated_loss = 0
        self.accumulated_valid_loss = 0

        self.train_data_loader = None
        self.valid_data_loader = None

        self.train_losses = []
        self.valid_losses = []

        self.save_best_only = save_best_only
        self.save_model_path = save_model_path
        self.save_loss_log = save_loss_log

        self.patient = patient
        self.patient_count = 0

        self.correct_num = 0
        self.accuracy = []

    def fit(self, epoch, train_data_loader, valid_data_loader):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.patient_count = 0
        self.correct_num = 0

        if not self.patient:
            self.patient = epoch

        self.n_minibatches = len(self.train_data_loader)
        print("total number of minibatches: {}".format(self.n_minibatches))

        for i in range(epoch):
            print("##### training epoch {}".format(i))

            train_loss = self.train()
            print("train loss: {}".format(train_loss))
            self.train_losses.append(train_loss)

            valid_loss, accuracy = self.valid()
            print("valid losss: {}".format(valid_loss))
            self.valid_losses.append(valid_loss)
            print("accuracy: {}".format(accuracy))
            self.accuracy.append(accuracy)

            if i > 0 and self.valid_losses[i - 1] > valid_loss:
                self.patient_count = 0
                if self.save_best_only:
                    self.save()
                    print("model saved")
            elif i != 0:
                self.patient_count += 1
                if self.patient_count > self.patient:
                    print("========= training aborted by early stopping ==========")
                    return

            if i == 0:
                self.save()
            elif not self.save_best_only:
                self.save()

        print("========== training finished =========")

    def train(self):
        self.model.train()

        self.accumulated_loss = 0
        for idx, (X, additional_X, Y) in tqdm(enumerate(self.train_data_loader)):
            self.train_minibatch(X, additional_X, Y)
        train_loss = self.accumulated_loss / len(self.train_data_loader)
        return train_loss

    def train_minibatch(self, X, additional_X, Y):
        X, additional_X, Y = self.load_batch(X, additional_X, Y)

        self.optimizer.zero_grad()
        predict_y = self.model(X, additional_X)
        # print(predict_y)
        # print(Y)
        loss = self.loss_function(predict_y, Y)

        self.accumulated_loss += loss.data[0]

        loss.backward()
        self.optimizer.step()

    def load_batch(self, X, additional_X, Y, is_train=True):
        if self.use_gpu and is_train:
            X = torch.autograd.Variable(X.cuda(), requires_grad=True)
            additional_X = torch.autograd.Variable(additional_X.cuda(),
                                                   requires_grad=True)
            Y = torch.autograd.Variable(Y.cuda())
        elif self.use_gpu:
            X = torch.autograd.Variable(X.cuda(), requires_grad=False, volatile=True)
            additional_X = torch.autograd.Variable(additional_X.cuda(), requires_grad=False,
                                                   volatile=True)
            Y = torch.autograd.Variable(Y.cuda(), requires_grad=False, volatile=True)
        elif is_train:
            X = torch.autograd.Variable(X, requires_grad=True)
            additional_X = torch.autograd.Variable(additional_X, requires_grad=True)
            Y = torch.autograd.Variable(Y)
        else:
            X = torch.autograd.Variable(X, requires_grad=False, volatile=True)
            additional_X = torch.autograd.Variable(additional_X, requires_grad=False,
                                                   volatile=True)
            Y = torch.autograd.Variable(Y, requires_grad=False, volatile=True)

        return X, additional_X, Y

    def valid(self):
        self.model.eval()
        self.accumulated_valid_loss = 0

        self.correct_num = 0

        for idx, (X, additional_X, Y) in tqdm(enumerate(self.valid_data_loader)):
            self.valid_minibatch(X, additional_X, Y)
        valid_loss = self.accumulated_valid_loss / len(self.valid_data_loader)

        accuracy = self.correct_num / len(self.valid_data_loader.dataset)
        return valid_loss, accuracy

    def valid_minibatch(self, X, additional_X, Y):
        X, additional_X, Y = self.load_batch(X, additional_X, Y, False)
        predict_Y = self.model(X, additional_X)
        loss = self.loss_function(predict_Y, Y)

        self.correct_num += self.is_correct(predict_Y, Y).sum().data[0]

        self.accumulated_valid_loss += loss.data[0]

    def is_correct(self, predict_Y, Y):
        return (predict_Y >= 0.5).float() == Y

    def save(self):
        torch.save(self.model.state_dict(), self.save_model_path)
        with self.save_loss_log.open(mode="w+") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(",".join(["train_losses", "valid_losses", "accuracy"]))
            for train_loss, valid_loss, acc in zip(self.train_losses, self.valid_losses, self.accuracy):
                writer.writerow([train_loss, valid_loss, acc])

