from curses.ascii import isalnum
import time
import uproot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

# for binary classification


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.fill_(0.01)
        self.linear1.bias.data.fill_(0.01)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.fill_(0.01)
        self.linear2.bias.data.fill_(0.01)
        self.linear3 = nn.Linear(hidden_size, num_class)
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
        out = self.relu(out)
        out = self.linear3(out)
        y_pred = self.sigmoid(out)
        #print(y_pred, end=" ")
        return y_pred


def get_data():
    filename1 = "/Users/ZOUXIANG/Desktop/CERN_Summer_Programme/tth_project/files/out_346345_p4498_mc16a.root"
    file1 = uproot.open(filename1)

    filename2 = "/Users/ZOUXIANG/Desktop/CERN_Summer_Programme/tth_project/files/out_410470_mc16a.root"
    file2 = uproot.open(filename2)

    tree1 = file1["nominal"]
    tree2 = file2["nominal"]
    # print(tree.keys())
    #keys1 = tree1.keys()
    #keys2 = tree2.keys()
    features = list()
    branches1 = tree1.arrays()
    branches2 = tree2.arrays()
    fin = open(
        "/Users/ZOUXIANG/Desktop/CERN_Summer_Programme/tth_project/Features/features.txt", "r")
    for x in fin:
        features.append(x.split()[0])
    fin.close()
    ttH = dict()
    tt = dict()
    for name in features:  # need to check whether it is vector!
        is_all_zero1 = np.all(np.array(branches1[name]) == 0)
        is_all_zero2 = np.all(np.array(branches2[name]) == 0)
        if is_all_zero1 and is_all_zero2:
            pass
        else:
            b1 = np.array(branches1[name], dtype=np.float32)
            b2 = np.array(branches2[name], dtype=np.float32)
            b1 = (b1-np.mean(b1))/np.std(b1)
            b2 = (b2-np.mean(b2))/np.std(b2)
            ttH[name] = b1
            tt[name] = b2
    print(len(ttH))

    train_data1 = np.zeros(shape=(15000, len(ttH)))  # out of 19782
    train_data2 = np.zeros(shape=(5000, len(ttH)))  # out of 6654
    test_data1 = np.zeros(shape=(19782-15000, len(ttH)))
    test_data2 = np.zeros(shape=(6654-5000, len(ttH)))

    for i in range(15000):
        j = 0
        for keys in ttH:
            if np.isnan(ttH[keys][i]):
                train_data1[i][j] = 0
            else:
                train_data1[i][j] = ttH[keys][i]
            j += 1
        #train_data1[i][train_data1[i] == 0] = 0.00001
    for i in range(5000):
        j = 0
        for keys in tt:
            if np.isnan(tt[keys][i]):
                train_data2[i][j] = 0
            else:
                train_data2[i][j] = tt[keys][i]
            j += 1
        #train_data2[i][train_data2[i] == 0] = 0.00001
    for i in range(15000, 19782):
        idx = i-15000
        j = 0
        for keys in ttH:
            if np.isnan(ttH[keys][i]):
                test_data1[idx][j] = 0
            else:
                test_data1[idx][j] = ttH[keys][i]
            j += 1
        #test_data1[idx][test_data1[idx] == 0] = 0.00001
    for i in range(5000, 6654):
        idx = i-5000
        j = 0
        for keys in tt:
            if np.isnan(tt[keys][i]):
                test_data2[idx][j] = 0
            else:
                test_data2[idx][j] = tt[keys][i]
            j += 1
        #test_data2[idx][test_data2[idx] == 0] = 0.0001
    # print(train_data1[3456])
    return (train_data1, train_data2, test_data1, test_data2, len(ttH))


def predict(insize, num_epoch, train_data1, train_data2, test_data1, test_data2):
    #train_data1, train_data2, test_data1, test_data2, insize = get_data()
    model = NeuralNet(insize, 25, 2)
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # num_epoch = 2  # can change
    '''
    tot_test_pos = 0
    correct_pos = 0
    tot_test_neg = 0
    correct_neg = 0

    for i in range(len(test_data1)):
        tot_test_pos += 1
        t = torch.from_numpy(test_data1[i])
        t = t.float()
        pred = model(t)
        #print(pred1.data[0], end=' ')
        if pred >= 0.5:
            correct_pos += 1

    print("for ttH dection: ", correct_pos, "/",
          tot_test_pos, " = ", correct_pos/tot_test_pos)

    for i in range(len(test_data2)):
        tot_test_neg += 1
        t = torch.from_numpy(test_data2[i])
        t = t.float()
        pred = model(t)
        if pred < 0.5:
            correct_neg += 1

    print("for not ttH detection: ", correct_neg, "/",
          tot_test_neg, " = ", correct_neg/tot_test_neg)
    '''
    time1 = time.time()
    for epoch in range(num_epoch):
        np.random.shuffle(train_data1)
        all_train_tot = np.concatenate(
            (train_data1[0: 5000], train_data2.copy()), axis=0)
        # print(all_train.shape)
        all_lab_tot = np.zeros(shape=(len(all_train_tot), 2))
        # print(all_lab.shape)
        for i in range(5000):
            all_lab_tot[i] = np.array([1, 0])
        for i in range(5000, len(all_train_tot)):
            all_lab_tot[i] = np.array([0, 1])
        idxs = np.arange(len(all_lab_tot))
        np.random.shuffle(idxs)
        print(idxs)
        all_train = all_train_tot[idxs]
        all_lab = all_lab_tot[idxs]
        for i in range(len(all_train)):
            t = torch.from_numpy(np.array(all_train[i]))
            t = t.float()
            # print(t.shape)
            pred = model(t)
            label = torch.tensor([all_lab[i]])
            loss = criterion(pred, label.float())

            optimizer.zero_grad()
            loss.backward()
            #print(t1.grad, end=", ")
            optimizer.step()
            if (i+1) % 1000 == 0:
                print(
                    f'epoch {epoch + 1} / {num_epoch}, step {i+1}/{len(all_train)}, loss = {loss.item():.8f}')

        #learning_rate = learning_rate * np.power(1/10, 1/num_epoch)
        #criterion = nn.BCELoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # the testing phase:
    time2 = time.time()
    train_time = time2-time1
    tot_test_pos = 0
    correct_pos = 0
    tot_test_neg = 0
    correct_neg = 0
    ttH_ratio = np.zeros(len(test_data1))
    tt_ratio = np.zeros(len(test_data2))
    for i in range(len(test_data1)):
        tot_test_pos += 1
        t1 = torch.from_numpy(test_data1[i])
        t1 = t1.float()
        pred1 = model(t1)
        if pred1[0] >= pred1[1]:
            correct_pos += 1
        ttH_ratio[i] = pred1[0]/pred1[1]

    print("for ttH dection: ", correct_pos, "/",
          tot_test_pos, " = ", correct_pos/tot_test_pos)

    for i in range(len(test_data2)):
        tot_test_neg += 1
        t1 = torch.from_numpy(test_data2[i])
        t1 = t1.float()
        pred1 = model(t1)
        if pred1[0] < pred1[1]:
            correct_neg += 1
        tt_ratio[i] = pred1[0]/pred1[1]

    print("for not ttH detection: "+str(correct_neg)+"/",
          str(tot_test_neg)+" = "+str(correct_neg/tot_test_neg))

    return(correct_pos, tot_test_pos, correct_neg, tot_test_neg, ttH_ratio, tt_ratio, train_time)


if __name__ == "__main__":
    train_data1, train_data2, test_data1, test_data2, insize = get_data()
    predict(3)
