import os
from Process.dataset import *
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Bitcoin":
        treePath = os.path.join(cwd,'data/Bitcoin/bitcointree.txt')
        print("reading Bitcoin tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2]), line.split(' ')[3:]
            #eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))

    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

#used：
def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + 'graph')

    #traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    traindata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)# no drop
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)

    return traindata_list, testdata_list

def loadBiData_a(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate,is_arguement):
    data_path = os.path.join(cwd,'data', dataname + 'graph')

    traindata_list = BiGraphDataset_a(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path,is_arguement=is_arguement)
    testdata_list = BiGraphDataset_a(fold_x_test, treeDic, data_path=data_path,is_arguement=is_arguement)

    return traindata_list, testdata_list



