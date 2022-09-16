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
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer


def get_data():
    """This function will return 2 dictionaries of features of simulations."""
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
            ttH[name] = b1
            tt[name] = b2

    print(len(ttH))

    return (ttH, tt)


def complete_graph(x):
    """This function generate a complete bi-directional graph of x nodes"""

    """
    x: int
        the number of nodes in the graph
    """
    u = []
    v = []
    for i in range(x-1):
        for j in range(i+1, x):
            u.append(i)
            v.append(j)
    src = np.array(u + v)
    dst = np.array(v + u)
    g = dgl.graph((src, dst))
    return g

# the edata can store the distance between the 2 node. (angular distance)


def graph_init():
    g = complete_graph(9)

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

    # data structure is np.array([Pt, Eta, Phi, E, Charge])

    j0 = np.array([simulation["jet_pt0"][i], simulation['jet_eta0']
                  [i], 0, 0, 0])
    j1 = np.array([simulation["jet_pt1"][i], simulation['jet_eta1']
                  [i], 0, 0, 0])
    j2 = np.array([simulation["jet_pt1"][i], simulation['jet_eta1']
                  [i], 0, 0, 0])
    if abs(simulation["lep_ID_0"][i]) == 11 and abs(simulation["lep_ID_1"][i]) == 11:
        e0 = np.array([simulation['lep_Pt_0'][i], simulation["lep_Eta_0"]
                      [i], simulation["lep_Phi_0"][i], simulation["lep_E_0"][i], -1])
        e1 = np.array([simulation['lep_Pt_1'][i], simulation["lep_Eta_1"]
                      [i], simulation["lep_Phi_1"][i], simulation["lep_E_1"][i], -1])
        m0 = np.array([0, 0, 0, 0, 0])
        m1 = np.array([0, 0, 0, 0, 0])
    elif abs(simulation["lep_ID_0"][i]) == 11 and abs(simulation["lep_ID_1"][i]) == 13:
        e0 = np.array([simulation['lep_Pt_0'][i], simulation["lep_Eta_0"]
                      [i], simulation["lep_Phi_0"][i], simulation["lep_E_0"][i], -1])
        e1 = np.array([0, 0, 0, 0, 0])
        m0 = np.array([simulation['lep_Pt_1'][i], simulation["lep_Eta_1"]
                      [i], simulation["lep_Phi_1"][i], simulation["lep_E_1"][i], -1])
        m1 = np.array([0, 0, 0, 0, 0])
    elif abs(simulation["lep_ID_0"][i]) == 13 and abs(simulation["lep_ID_1"][i]) == 11:
        e0 = np.array([simulation['lep_Pt_1'][i], simulation["lep_Eta_1"]
                      [i], simulation["lep_Phi_1"][i], simulation["lep_E_1"][i], -1])
        e1 = np.array([0, 0, 0, 0, 0])
        m0 = np.array([simulation['lep_Pt_0'][i], simulation["lep_Eta_0"]
                      [i], simulation["lep_Phi_0"][i], simulation["lep_E_0"][i], -1])
        m1 = np.array([0, 0, 0, 0, 0])
    elif abs(simulation["lep_ID_0"][i]) == 13 and abs(simulation["lep_ID_1"][i]) == 13:
        m0 = np.array([simulation['lep_Pt_0'][i], simulation["lep_Eta_0"]
                      [i], simulation["lep_Phi_0"][i], simulation["lep_E_0"][i], -1])
        m1 = np.array([simulation['lep_Pt_1'][i], simulation["lep_Eta_1"]
                      [i], simulation["lep_Phi_1"][i], simulation["lep_E_1"][i], -1])
        e0 = np.array([0, 0, 0, 0, 0])
        e1 = np.array([0, 0, 0, 0, 0])
    t0 = np.array([simulation["taus_pt_0"][i], simulation["taus_eta_0"][i],
                  simulation["taus_phi_0"][i], 0, simulation["taus_charge_0"][i]])
    t1 = np.array([simulation["taus_pt_1"][i], simulation["taus_eta_1"][i],
                  simulation["taus_phi_1"][i], 0, simulation["taus_charge_1"][i]])
    g.ndata["features"] = torch.from_numpy(
        np.array([j0, j1, j2, e0, e1, m0, m1, t0, t1]))


def prep_train_graphlist(simulation, is_ttH, size=5000):
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
    graph_list = []
    for i in range(size):
        g = graph_init()
        assign_feature(g, simulation, i)
        graph_list.append([g, int(is_ttH)])

    return graph_list


def prep_test_graphlist(simulation, is_ttH, size=1500):
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
    for i in range(5000, 5000+size):  # just for testing
        g = graph_init()
        assign_feature(g, simulation, i)
        graph_list.append([g, int(is_ttH)])

    return graph_list


def get_tot_train_graph(graphlist):
    """This function will take a graphlist and then generate the batched graph and the label"""

    """
     graphlist should be a list of items as follows:
        [dgl.graph, int] (a graph object and a label)
    """

    index_for_shuffle = np.arange(len(graphlist))
    np.random.shuffle(index_for_shuffle)
    x = 0
    labels = np.zeros(len(graphlist))
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
    return batched_g, labels


def train(model, graphlist, num_epoch):
    """This function will train the model with graph data"""

    """
    model: nn.Moduler
        the model to be trained
    tot_train: dgl.graph
        the batched graph that contains all graphs to be trained
    labels: numpy 1D array
        the labels for individual graph used for supervised learning
    num_epoch: int
        the number of epoch for training
    """
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(num_epoch):
        if epoch % 100 == 0:
            tot_train, labels = get_tot_train_graph(graphlist)
        features = tot_train.ndata['features']
        features = features.float()
        logits = model(tot_train, features)
        logits = logits.float()
        train_labels = np.zeros(shape=(len(labels), 2))
        for i in range(len(labels)):
            if labels[i] == 1:
                train_labels[i] = np.array([1, 0])
            else:
                train_labels[i] = np.array([0, 1])

        train_labels = torch.from_numpy(train_labels).float()
        loss = F.cross_entropy(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f'epoch {epoch + 1} / {num_epoch}, loss = {loss.item():.8f}')

    return model


def test(model, ttH, tt):
    """This function will test the performance of the model"""

    """
    model: nn.Moduler 
        the model to be tested
    ttH: dictionary
        the dictionary containing ttH events features
    tt: dictionary
        the dictionary containing tt events features
    """

    # tesing phase:
    ttH_test_glist = prep_test_graphlist(ttH, True, 1500)
    tt_test_glist = prep_test_graphlist(tt, False, 1500)
    true_pos = 0
    tot_pos = 0
    true_neg = 0
    tot_neg = 0
    for i in range(1500):
        features = ttH_test_glist[i][0].ndata['features']
        features = features.float()
        logits = model(ttH_test_glist[i][0], features)
        logits = logits.float()
        #this_label = torch.from_numpy(np.array([ttH_test_labels[i]])).long()
        #loss = F.cross_entropy(logits, this_label)
        tot_pos += 1
        if logits[0][0] >= logits[0][1]:
            true_pos += 1
    print("true positive rate is ", true_pos,
          " / ", tot_pos, " = ", true_pos/tot_pos)

    for i in range(1500):
        features = tt_test_glist[i][0].ndata['features']
        features = features.float()
        logits = model(tt_test_glist[i][0], features)
        #this_label = torch.from_numpy(np.array([tt_test_labels[i]])).long()
        #loss = F.cross_entropy(logits, this_label)
        tot_neg += 1
        if logits[0][0] < logits[0][1]:
            true_neg += 1
    print("true negative rate is ", true_neg,
          " / ", tot_neg, " = ", true_neg/tot_neg)

    return true_pos, tot_pos, true_neg, tot_neg


def main():
    start_time = time.time()
    ttH, tt = get_data()
    time1 = time.time()
    print("finish getting data, using ", time1-start_time, "s")
    ttH_train = prep_train_graphlist(ttH, True, 5000)
    time2 = time.time()
    print(len(ttH_train), " using ", time2-time1, "s")
    tt_train = prep_train_graphlist(tt, False, 5000)
    time3 = time.time()
    print(len(ttH_train), " using ", time3-time2, "s")
    # Only an example, 5 is the input feature size
    model = GIN(3, 3, 5, 10, 2, 0.1, False, "mean", "mean")
    test(model, ttH, tt)
    all_list = ttH_train + tt_train
    outputfile = "dgl_testing.txt"
    for i in range(1000):
        fin = open(outputfile, "a")
        model = train(model, all_list, 1000)
        true_pos, tot_pos, true_neg, tot_neg = test(model, ttH, tt)
        fin.write("for ttH detection: "+str(true_pos)+"/" +
                  str(tot_pos)+" = "+str(true_pos/tot_pos)+"\t")
        fin.write("for not ttH detection: "+str(true_neg)+"/" +
                  str(tot_neg)+" = "+str(true_neg/tot_neg)+"\n")
        fin.close()


if __name__ == "__main__":
    main()
