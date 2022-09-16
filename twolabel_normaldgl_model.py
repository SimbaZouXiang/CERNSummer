from random import shuffle
import time
import uproot
import dgl
import torch
import dgl.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import dgl.nn.pytorch as dglnn

"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.sageconv1 = dglnn.SAGEConv(
            hidden_dim, hidden_dim, aggregator_type='mean')
        self.sageconv2 = dglnn.SAGEConv(
            hidden_dim, hidden_dim, aggregator_type='gcn')
        # self.chebconv = dglnn.ChebConv(hidden_dim, hidden_dim, 2)
        # self.gatconv1 = dglnn.GATConv(hidden_dim, hidden_dim, 1)
        self.sageconv3 = dglnn.SAGEConv(
            hidden_dim, hidden_dim, aggregator_type='mean')
        self.sageconv4 = dglnn.SAGEConv(
            hidden_dim, hidden_dim, aggregator_type='mean')
        # self.relgraphconv = dglnn.RelGraphConv(hidden_dim, hidden_dim, 3, regularizer='basis', num_bases=2, )
        # self.gatedconv = dglnn.GatedGraphConv(hidden_dim, hidden_dim, 10, 1)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.sageconv1(g, h))
        # h = F.relu(self.chebconv(g, h))
        # h = F.relu(self.gatconv1(g, h))
        # h = F.relu(self.relgraphconv(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.sageconv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.sageconv3(g, h))
        '''
        h = F.relu(self.conv4(g, h))
        h = F.relu(self.sageconv4(g, h))
        # h = F.relu(self.gatedconv(g, h))
        '''
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


def get_data():
    """This function will return 2 dictionaries of features of simulations."""
    filename1 = "/Users/ZOUXIANG/Desktop/CERN_Summer_Programme/tth_project/files/out_346345_p4498_mc16a.root"
    file1 = uproot.open(filename1)

    filename2 = "/Users/ZOUXIANG/Desktop/CERN_Summer_Programme/tth_project/files/out_410470_mc16a.root"
    file2 = uproot.open(filename2)

    tree1 = file1["nominal"]
    tree2 = file2["nominal"]
    # print(tree.keys())
    # keys1 = tree1.keys()
    # keys2 = tree2.keys()
    features = list()
    branches1 = tree1.arrays()
    branches2 = tree2.arrays()
    fin = open(
        "/Users/ZOUXIANG/Desktop/CERN_Summer_Programme/tth_project/Features/dgl_features.txt", "r")
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

            b_tot = np.concatenate((b1, b2, b2, b2), axis=0)
            b1 = (b1 - np.mean(b_tot))/np.std(b_tot)
            b2 = (b2 - np.mean(b_tot))/np.std(b_tot)
            ttH[name] = b1
            tt[name] = b2
            '''
            '''

    print(len(ttH))

    return (ttH, tt)


def complete_graph(start_idx, tot_nodes):
    """This function generate a complete bi-directional graph of n nodes"""

    """
    start_idx: int
        the starting node index
    tot_nodes:
        the total number of nodes to be generated
    """
    u = []
    v = []
    for i in range(start_idx, start_idx + tot_nodes - 1):
        for j in range(i+1, start_idx + tot_nodes):
            u.append(i)
            v.append(j)
    src = u + v
    dst = v + u
    return src, dst

# the edata can store the distance between the 2 node. (angular distance)


def graph_init(jet_num, par_num):

    src1, dst1 = complete_graph(0, jet_num)
    src2, dst2 = complete_graph(jet_num, par_num)
    all_src = np.array(src1 + src2 + [0, jet_num+par_num - 1])
    all_dst = np.array(dst1 + dst2 + [jet_num+par_num - 1, 0])
    g = dgl.graph((all_src, all_dst))

    '''
    node 0: jet 0
    node 1: jet 1
    node 2: jet 2
    node 3: electron 0
    node 4: electron 1
    node 5: muon 0
    node 6: muon 1
    node 7: tau 0
    node 8: tau 1
    '''
    return g


def assign_feature(g, simulation, i):
    """This function assigned the features to the the graph g"""
    """
    g: DGL graph
        the DGL graph that you want to assign feature to
    simulation: dictionary-like object
        the dictionary for features that you want to assign
    i: int
        the i-th event in the feature dictionary that you want to creat the graph

    """

    # data structure is np.array([Pt, Eta, Phi, E, ID])

    """
    Pt, Eta, Phi, E are in monte-carlo simulation units
    Charge is in usual charge unit
    mass is in MeV
    """
    j0 = np.array([simulation["jet_pt0"][i], simulation['jet_eta0']
                  [i], 0, 0, 0])
    j1 = np.array([simulation["jet_pt1"][i], simulation['jet_eta1']
                  [i], 0, 0, 0])
    j2 = np.array([simulation["jet_pt1"][i], simulation['jet_eta1']
                  [i], 0, 0, 0])
    bj = np.array([simulation["bjet_pt0"][i], simulation['bjet_eta0']
                  [i], 0, 0, 0])
    l0_ID = simulation["lep_ID_0"][i]
    l1_ID = simulation["lep_ID_1"][i]

    l0 = np.array([simulation['lep_Pt_0'][i], simulation["lep_Eta_0"]
                   [i], simulation["lep_Phi_0"][i], simulation["lep_E_0"][i], l0_ID])
    l1 = np.array([simulation['lep_Pt_1'][i], simulation["lep_Eta_1"]
                   [i], simulation["lep_Phi_1"][i], simulation["lep_E_1"][i], l1_ID])

    tauon_charge_0 = simulation["taus_charge_0"][i]
    tauon_charge_1 = simulation["taus_charge_1"][i]

    t0 = np.array([simulation["taus_pt_0"][i], simulation["taus_eta_0"][i],
                  simulation["taus_phi_0"][i], 0, tauon_charge_0*(1.25)])
    t1 = np.array([simulation["taus_pt_1"][i], simulation["taus_eta_1"][i],
                  simulation["taus_phi_1"][i], 0, tauon_charge_1*(1.25)])
    # nutrino = np.array([simulation["met_met"][i], 0,
    #                    simulation["met_phi"][i], 0, 0])
    g.ndata["features"] = torch.from_numpy(
        np.array([j0, j1, j2, bj, l0, l1, t0, t1]))
    # np.array([l0, l1, t0, t1]))


def prep_train_graphlist(simulation, is_ttH, size):
    """This function will prepare a list of graph for training"""

    """
    simulation: dictionary
        either ttH or tt, a dictionary with all features.
    is_ttH: bool
        whether the dictionary represent ttH result.
    size: int
        the size of the graph to be generated
    """
    # for i in range(len(simulation["jet_pt1"])):
    if is_ttH:
        graph_list = []
        deck = np.array(range(1, 15000))
        np.random.shuffle(deck)
        for x in range(size):
            i = deck[x]
            g = graph_init(4, 4)
            assign_feature(g, simulation, i)
            graph_list.append([g, int(is_ttH)])
    else:
        graph_list = []
        for i in range(size):
            g = graph_init(4, 4)
            assign_feature(g, simulation, i)
            graph_list.append([g, int(is_ttH)])

    return graph_list


def prep_test_graphlist(simulation, is_ttH, size, start_from):
    """this function will return a list of DGL graphs for testing"""

    """
    simulation: dictionary
        either ttH or tt, a dictionary with all features.
    is_ttH: bool
        whether the dictionary represent ttH result.
    size: int
        the size of the graph to be generated
    """

    graph_list = []
    for i in range(start_from, start_from+size):  # just for testing
        g = graph_init(4, 4)
        assign_feature(g, simulation, i)
        graph_list.append([g, int(is_ttH)])

    return graph_list


def get_tot_graph(graphlist):
    """This function will take a graphlist and then generate the batched graph and the label"""

    """
     graphlist should be a list of items as follows:
        [dgl.graph, int] (a graph object and a label)
    """

    index_for_shuffle = np.arange(len(graphlist))
    np.random.shuffle(index_for_shuffle)
    x = 0
    labels = np.zeros(len(graphlist))
    time1 = time.time()
    for i in index_for_shuffle:
        if x == 0:
            batched_g = graphlist[i][0]
            labels[x] = graphlist[i][1]
            x += 1
        else:
            g = graphlist[i][0]
            batched_g = dgl.batch([batched_g, g])
            labels[x] = graphlist[i][1]
            x += 1
    time2 = time.time()
    print("number of nodes = ", batched_g.number_of_nodes(),
          " using ", time2 - time1)
    return batched_g, labels


def train(model, graphlist, num_epoch):
    """This function will train the model with graph data"""

    """
    model: nn.Moduler
        the model to be trained

    graphlist: list object
        the element of the list is in the format of [dgl.Graph, int]
    tot_train: dgl.graph
        the batched graph that contains all graphs to be trained
    labels: numpy 1D array
        the labels for individual graph used for supervised learning
    num_epoch: int
        the number of epoch for training
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epoch):
        if epoch % 25 == 0:
            tot_train, labels = get_tot_graph(graphlist)
        features = tot_train.ndata['features']
        features = features.float()
        logits = model(tot_train, features)
        logits = logits.float()
        train_labels = np.zeros(shape=(len(labels), 2))
        for i in range(len(labels)):
            if labels[i] == 1:
                train_labels[i] = np.array([1, 0])  # np.array([1])
            else:
                train_labels[i] = np.array([0, 1])  # np.array([0])

        train_labels = torch.from_numpy(train_labels).float()
        # this line is changed!!!!!!!!!!!!!!!!!!!
        # print(logits, len(logits))
        # print(train_labels, len(train_labels))
        try:
            logits = logits[:, -1, :]
        except:
            pass
        loss = F.mse_loss(logits, train_labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f'epoch {epoch + 1} / {num_epoch}, loss = {loss.item():.8f}')

    return model


def test(model, ttH_test_glist, tt_test_glist):
    """This function will test the performance of the model"""

    """
    model: nn.Moduler 
        the model to be tested
    ttH_tot_graph: dgl.Graph
        the total batched graph consisting all ttH testing sample
    tt_tot_graph: dgl.Graph
        the total batched graph consisting all tt testing sample
    """

    # tesing phase:

    true_pos = 0
    tot_pos = 0
    true_neg = 0
    tot_neg = 0
    ttH_ratio_list = np.zeros(len(ttH_test_glist))
    tt_ratio_list = np.zeros(len(tt_test_glist))

    for i in range(len(ttH_test_glist)):
        features = ttH_test_glist[i][0].ndata['features']
        features = features.float()
        logits1 = model(ttH_test_glist[i][0], features)
        logits1 = logits1.float()
        logits1 = logits1.detach().numpy()
        # this_label = torch.from_numpy(np.array([ttH_test_labels[i]])).long()
        # loss = F.cross_entropy(logits, this_label)
        tot_pos += 1
        ttH_ratio_list[i] = logits1[0][0]-logits1[0][1]
        # print(logits)  # check the output
        if logits1[0][0] >= logits1[0][1]:
            true_pos += 1

    '''
    features = ttH_tot_graph.ndata['features']
    features = features.float()
    logits = model(ttH_tot_graph, features)
    logits = logits.float()
    print(logits)
    true_labels = np.zeros(shape=len_ttH)
    false_labels = np.zeros(shape=len_ttH)
    for i in range(len_ttH):
        true_labels[i] = np.array([1])
        false_labels[i] = np.array([0])
    true_labels = torch.from_numpy(true_labels).long()
    false_labels = torch.from_numpy(false_labels).long()
    loss1 = F.cross_entropy(logits, true_labels)
    loss2 = F.cross_entropy(logits, false_labels)
    loss1 = loss1.detach().numpy()
    print(loss1)
    loss2 = loss2.detach().numpy()
    true_pos = np.sum(loss1 <= loss2)
    '''
    print("true positive rate is ", true_pos,
          " / ", tot_pos, " = ", true_pos/tot_pos)

    for i in range(len(tt_test_glist)):
        features = tt_test_glist[i][0].ndata['features']
        features = features.float()
        logits2 = model(tt_test_glist[i][0], features)
        logits2 = logits2.detach().numpy()
        tt_ratio_list[i] = logits2[0][0]-logits2[0][1]
        # this_label = torch.from_numpy(np.array([tt_test_labels[i]])).long()
        # loss = F.cross_entropy(logits, this_label)
        tot_neg += 1
        if logits2[0][0] < logits2[0][1]:
            true_neg += 1
    '''
    features = tt_tot_graph.ndata['features']
    features = features.float()
    logits = model(tt_tot_graph, features)
    logits = logits.float()
    true_labels = torch.from_numpy(np.ones(len_tt)).long()
    false_labels = torch.from_numpy(np.zeros(len_tt)).long()
    loss1 = F.cross_entropy(logits, true_labels)
    loss2 = F.cross_entropy(logits, false_labels)
    loss1 = loss1.detach().numpy()
    loss2 = loss2.detach().numpy()
    true_neg = np.sum(loss1 > loss2)
    '''
    print("true negative rate is ", true_neg,
          " / ", tot_neg, " = ", true_neg/tot_neg)

    return true_pos, tot_pos, true_neg, tot_neg, ttH_ratio_list, tt_ratio_list


def main():
    x = graph_init(4, 4)
    print(x.number_of_nodes())
    start_time = time.time()
    ttH, tt = get_data()
    time1 = time.time()
    print("finish getting data, using ", time1-start_time, "s")
    ttH_train = prep_train_graphlist(ttH, True, 5000)
    time2 = time.time()
    print(len(ttH_train), " using ", time2-time1, "s")
    tt_train = prep_train_graphlist(tt, False, 5000)
    time3 = time.time()
    print(len(tt_train), " using ", time3-time2, "s")
    # Only an example, 5 is the input feature size
    ttH_test_glist = prep_test_graphlist(ttH, True, 4781, 15000)
    tt_test_glist = prep_test_graphlist(tt, False, 1653, 5000)
    #test(model, ttH_test_glist, tt_test_glist)
    all_list = ttH_train + tt_train
    for run in range(1000):
        model = Classifier(5, 20, 2)
        previous_time = 0
        '''
        fin1 = open("dgl_accuracy.txt", "a")
        fin2 = open("dgl_ttH_.txt", "a")
        fin3 = open("dgl_tt_ratio.txt", "a")
        fin1.write("\n this is run " + str(run) + "\n")
        fin2.write("\n this is run " + str(run) + "\n")
        fin3.write("\n this is run " + str(run) + "\n")
        fin1.close()
        fin2.close()
        fin3.close()
        '''
        for i in range(6):
            ttH_train = prep_train_graphlist(ttH, True, 5000)
            tt_train = prep_train_graphlist(tt, False, 5000)
            all_list = ttH_train + tt_train
            time1 = time.time()
            model = train(model, all_list, 500)
            time2 = time.time()
            true_pos, tot_pos, true_neg, tot_neg, ratio_ttH, ratio_tt = test(
                model, ttH_test_glist, tt_test_glist)
            true_pos_rate = true_pos/tot_pos
            true_neg_rate = true_neg/tot_neg
            fin1 = open("dgl_accuracy.txt", "a")
            fin2 = open("dgl_ttH_minus3.txt", "a")
            fin3 = open("dgl_tt_minus3.txt", "a")
            np.savetxt(fin1, np.array(
                [time2-time1+previous_time, true_pos_rate, true_neg_rate]), newline=',')
            fin1.write("\n")
            np.savetxt(fin2, np.array(
                ratio_ttH), newline=',')
            fin2.write("\n")
            np.savetxt(fin3, np.array(
                ratio_tt), newline=',')
            fin3.write("\n")
            fin1.close()
            fin2.close()
            fin3.close()
            previous_time = time2 - time1


if __name__ == "__main__":
    main()
