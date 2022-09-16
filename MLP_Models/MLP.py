from curses.ascii import isalnum
import uproot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

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
            ttH[name] = np.array(branches1[name], dtype=np.float32)
            tt[name] = np.array(branches2[name], dtype=np.float32)
    print(len(ttH))
    train_data1 = np.zeros(shape=(15000, len(ttH)))  # out of 19782
    train_data2 = np.zeros(shape=(5000, len(ttH)))  # out of 6654
    test_data1 = np.zeros(shape=(19782-15000, len(ttH)))
    test_data2 = np.zeros(shape=(6654-5000, len(ttH)))

    for i in range(15000):
        j = 0
        for keys in ttH:
            if ttH[keys][i] > 0:
                train_data1[i][j] = np.log10(ttH[keys][i])
            elif ttH[keys][i] < 0:
                train_data1[i][j] = -np.log10(-ttH[keys][i])
            else:
                train_data1[i][j] = ttH[keys][i]
            j += 1
        #train_data1[i][train_data1[i] == 0] = 0.00001
    for i in range(5000):
        j = 0
        for keys in tt:
            if tt[keys][i] > 0:
                train_data2[i][j] = np.log10(tt[keys][i])
            elif tt[keys][i] < 0:
                train_data2[i][j] = -np.log10(-tt[keys][i])
            else:
                train_data2[i][j] = tt[keys][i]
            j += 1
        #train_data2[i][train_data2[i] == 0] = 0.00001
    for i in range(15000, 19782):
        idx = i-15000
        j = 0
        for keys in ttH:
            if ttH[keys][i] > 0:
                test_data1[idx][j] = np.log10(ttH[keys][i])
            elif ttH[keys][i] < 0:
                test_data1[idx][j] = -np.log10(-ttH[keys][i])
            else:
                test_data1[idx][j] = ttH[keys][i]
            j += 1
        #test_data1[idx][test_data1[idx] == 0] = 0.00001
    for i in range(5000, 6654):
        idx = i-5000
        j = 0
        for keys in tt:
            if tt[keys][i] > 0:
                test_data2[idx][j] = np.log10(tt[keys][i])
            elif tt[keys][i] < 0:
                test_data2[idx][j] = -np.log10(-tt[keys][i])
            else:
                test_data2[idx][j] = tt[keys][i]
            j += 1
        #test_data2[idx][test_data2[idx] == 0] = 0.0001
    # print(train_data1[3456])
    return (train_data1, train_data2, test_data1, test_data2, len(ttH))


def predict(model, num_epoch, train_data1, train_data2, test_data1, test_data2):
    #train_data1, train_data2, test_data1, test_data2, insize = get_data()
    #model = NeuralNet(input_size=insize, hidden_size=50)
    learning_rate = 0.00001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # num_epoch = 2  # can change
    all_train_tot = np.concatenate(
        (train_data1, train_data2.copy(), train_data2.copy(), train_data2.copy()), axis=0)
    # print(all_train.shape)
    all_lab_tot = np.zeros(shape=(len(all_train_tot)))
    # print(all_lab.shape)
    for i in range(len(train_data1)):
        all_lab_tot[i] = 1
    for i in range(len(train_data1), len(all_train_tot)):
        all_lab_tot[i] = 0

    tot_test_pos = 0
    correct_pos = 0
    tot_test_neg = 0
    correct_neg = 0

    for i in range(4781):
        tot_test_pos += 1
        t = torch.from_numpy(test_data1[i])
        t = t.float()
        pred = model(t)
        #print(pred1.data[0], end=' ')
        if pred >= 0.5:
            correct_pos += 1

    print("for ttH dection: ", correct_pos, "/",
          tot_test_pos, " = ", correct_pos/tot_test_pos)

    for i in range(1653):
        tot_test_neg += 1
        t = torch.from_numpy(test_data2[i])
        t = t.float()
        pred = model(t)
        if pred < 0.5:
            correct_neg += 1

    print("for not ttH detection: ", correct_neg, "/",
          tot_test_neg, " = ", correct_neg/tot_test_neg)

    for epoch in range(num_epoch):
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
    # the testing phase:
    tot_test_pos = 0
    correct_pos = 0
    tot_test_neg = 0
    correct_neg = 0
    predttH_all = np.zeros(4781)
    predtt_all = np.zeros(1653)
    for i in range(4781):
        tot_test_pos += 1
        t1 = torch.from_numpy(test_data1[i])
        t1 = t1.float()
        pred1 = model(t1)
        predttH_all[i] = pred1.data
        if pred1 >= 0.5:
            correct_pos += 1

    print("for ttH dection: ", correct_pos, "/",
          tot_test_pos, " = ", correct_pos/tot_test_pos)

    for i in range(1653):
        tot_test_neg += 1
        t2 = torch.from_numpy(test_data2[i])
        t2 = t2.float()
        pred2 = model(t2)
        predtt_all[i] = pred2.data
        if pred2 < 0.5:
            correct_neg += 1

    print("for not ttH detection: "+str(correct_neg)+"/",
          str(tot_test_neg)+" = "+str(correct_neg/tot_test_neg))

    return(correct_pos, tot_test_pos, correct_neg, tot_test_neg, predttH_all, predtt_all)


if __name__ == "__main__":
    train_data1, train_data2, test_data1, test_data2, insize = get_data()
    predict(3)
