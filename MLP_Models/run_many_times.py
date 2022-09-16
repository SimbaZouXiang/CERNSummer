from curses.ascii import isalnum
import uproot
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import modified_analysis as ma  # need to change everytime

# for binary classification


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.fill_(0.01)
        self.linear1.bias.data.fill_(0.01)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.fill_(0.01)
        self.linear2.bias.data.fill_(0.01)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.fill_(0.01)
        self.linear3.bias.data.fill_(0.01)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        # or we have the following choices

        # nn.Sigmoid
        # nn.Softmax
        # nn.Tanh
        # nn.LeakyReLU

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        #out = self.softmax(out)
        # print(out)
        out = self.linear2(out)
        # sigmoid at the end
        out = self.leakyrelu(out)
        out = self.linear3(out)
        y_pred = self.sigmoid(out)
        #print(y_pred, end=" ")
        return y_pred


def get_data():
    return ma.get_data()


train_data1, train_data2, test_data1, test_data2, insize = get_data()
time_start = time.time()
for i in range(10):
    fin1 = open("accuracy_new.txt", "a")
    fin1.write("This is run " + str(i) + "\n")
    fin1.close()
    model = NeuralNet(input_size=insize, hidden_size=50)
    for i in range(50):
        correct_pos, tot_test_pos, correct_neg, tot_test_neg, ttH_minus, tt_minus = ma.predict(
            model, 1, train_data1, train_data2, test_data1, test_data2)
        '''
        time2 = time.time()
        train_time = time2 - time_start
        time_start = time2
        '''
        fin1 = open("accuracy_new.txt", "a")
        true_pos_rate = correct_pos/tot_test_pos
        true_neg_rate = correct_neg/tot_test_neg
        np.savetxt(fin1, np.array(
            [true_pos_rate, true_neg_rate]), newline=',')
        fin1.write("\n")
        fin1.close()
    fin2 = open("ttH_minus.txt", "a")
    fin3 = open("tt_minus.txt", "a")
    np.savetxt(fin2, np.array(ttH_minus), newline=',')
    fin2.write("\n")
    np.savetxt(fin3, np.array(tt_minus), newline=',')
    fin3.write("\n")
    fin2.close()
    fin3.close()
