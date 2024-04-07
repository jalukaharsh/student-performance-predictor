from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        out = inputs
        sig = nn.Sigmoid()
        tanh = nn.Tanh()
        m = nn.Sequential(nn.Dropout(p=0.2), self.g, sig, self.h, sig)
        out = m(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: 3-tuple containing arrays for validation accuracies, training costs and corresponding epochs
    """

    valid_array = []
    train_array = []
    epoch_array = []
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            # target[0][nan_mask] = output[0][nan_mask]
            target[0:1][nan_mask] = output[0:1][nan_mask]

            loss = torch.sum((output - target) ** 2.) + lamb / 2 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        valid_array.append(valid_acc)
        train_array.append(train_loss)
        epoch_array.append(epoch)

    return valid_array, train_array, epoch_array

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def plot_graphs(epoch_array, k, lamb, loss_array, lr, valid_acc):
    title = 'k{}_lr({})_lamb({})_plots'.format(k, '_'.join(str(lr).split('.')), '_'.join(str(lamb).split('.')))
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1), plt.plot(epoch_array, loss_array, label='k = {}, lr = {}'.format(k, lr), color='#327ce3')
    plt.xlabel("Epoch"), plt.ylabel("Training Cost")
    plt.title("Training cost vs epoch")
    plt.legend()
    plt.subplot(1, 2, 2), plt.plot(epoch_array, valid_acc, label='k = {}, lr = {}'.format(k, lr), color='orange')
    plt.xlabel("Epoch"), plt.ylabel("Validation Accuracy")
    plt.title("Validation accuracy vs epoch")
    plt.legend()
    plt.savefig(title)


def main():
    path_harsh = "C:/Users/Harsh Jaluka/OneDrive/Desktop/CSC311 - Introduction to Machine Learning/Project/" \
                 "csc311-project/data"
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.

    # possible k = {10, 50, 100, 200, 500}
    # num_question = number of columns in train_matrix

    k = 115
    model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

    # Set optimization hyperparameters.
    # We worked with lr = {0.001, 0.01, 0.1, 1}
    lr = 0.01
    num_epoch = 40

    # lambda choices = {0.001, 0.01, 0.1, 1}
    lamb = 0.001

    print('lambda:', lamb, 'lr:', lr, 'k:', k)

    valid_acc, loss_array, epoch_array = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    plot_graphs(epoch_array, k, lamb, loss_array, lr, valid_acc)
    print(evaluate(model, zero_train_matrix, test_data))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":

    main()
