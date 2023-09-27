# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import copy
cwd=os.getcwd()
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = [] #对象
        self.idx = idx #current index
        #self.word = []
        #self.index = []
        self.feat=[]
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index-1)
    return wordFreq, wordIndex

def constructMat(tree): #tree=treeDic[eid]
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i) #Node_tweet对象
        index2node[i] = node
    for j in tree:
        indexC = j #当前index
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]#Node_tweet对象
        feat=list(map(float,tree[j]['vec']))#特征
        
        #wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        #nodeC.index = wordIndex #wordindex
        #nodeC.word = wordFreq #wordfreq
        nodeC.feat=feat
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
            rootindex=indexC-1
            root_feat=nodeC.feat
            #root_index=nodeC.index
            #root_word=nodeC.word
    #rootfeat = np.zeros([1, 5000])
    #if len(root_index)>0:
    #    rootfeat[0, np.array(root_index)] = np.array(root_word) #rootfeat[0,index]=word(词语->词频)
    ## 3. convert tree to matrix and edgematrix
    matrix=np.zeros([len(index2node),len(index2node)])
    raw=[]
    col=[]
    x_feat=[]
    #x_word=[]
    #x_index=[]
    edgematrix=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                raw.append(index_i)
                col.append(index_j)
        x_feat.append(index2node[index_i+1].feat)
        #x_word.append(index2node[index_i+1].word)
        #x_index.append(index2node[index_i+1].index)
    edgematrix.append(raw)
    edgematrix.append(col)
    return x_feat,edgematrix,root_feat,rootindex
    #return x_word, x_index, edgematrix,rootfeat,rootindex

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

# def node2index(event):
#     event2index={}
#     s1=[]
#     index={}
#     for i in event:
#         s1.append(i)
#         if event[i]['parent']!='None':
#             s1.append(int(event[i]['parent']))
#     s=list(set(s1))
#     s.sort(key=s1.index)
#     for i in range(len(s)):
#         index[s[i]]=i+1
#     for i in event:
#         if event[i]['parent']!='None':
#             event[i]['parent']=str(index[int(event[i]['parent'])])
#         event2index[index[i]]=event[i]
#     return event2index

def node2index(event):
    event2index={}
    s=[]
    index={}
    newevent=copy.deepcopy(event)
    #删除父节点
    for i in event:
        if event[i]['parent']!='None':
            if int(event[i]['parent']) not in event.keys():
                del newevent[i]
    
    for i in newevent:
        s.append(i)
        # if event[i]['parent']!='None':
        #     s1.append(int(event[i]['parent']))

    for i in range(len(s)):
        index[s[i]]=i+1
    for i in newevent:
        if newevent[i]['parent']!='None':
            newevent[i]['parent']=str(index[int(newevent[i]['parent'])])
        event2index[index[i]]=newevent[i]
    return event2index

def main():
    Path='/home/catlab/oysy/bitcoin_graph/Bit-CHetG-master'
    # treePath = os.path.join(Path, 'data/Bitcoin/elliptictree_5_01T.txt')
    treePath = os.path.join(Path, 'data/Bitcoin/bitcointree.txt')
    print("reading Btcoin tree")
    treeDic = {} # treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    

    print('tree no:', len(treeDic)) 
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC,Vec = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2]), line.split(' ')[3:]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    print('tree no:', len(treeDic)) #谣言树字典treeDic 14595

    # labelPath = os.path.join(Path, "data/Bitcoin/elliptic_5_01T_id_label.txt")
    labelPath = os.path.join(Path, "data/Bitcoin/bitcoin_id_label.txt")
    print("loading bitcoin label:")
    event,y= [],[]
    l1 = l2 = 0
    labelDic = {}# labelDic[event]=y
    for line in open(labelPath):
        line = line.rstrip()
        eid,label = line.split(' ')[0], line.split(' ')[1]
        # labelDic[eid] = int(label)
        #含非法率的二分类
        if round(float(label),3)<0.5:
            labelDic[eid] = 1
            y.append(labelDic[eid])
            event.append(eid)
            l1 += 1
        if round(float(label),3)>=0.5:
            labelDic[eid] = 0
            y.append(labelDic[eid])
            event.append(eid)
            l2 += 1

    print(len(labelDic),len(event),len(y))#labelDic 4664
    print(l1, l2)#0，1

    def loadEid(event,id,y): #loadEid(treeDic[eid],eid,labelDic[eid])
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            # x_word：词语频率
            # x_index: 词语编号
            # tree：边的邻接矩阵 行列编号 2*
            # rootfeat:1*5000,词频
            # rootindex：0
            #x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            #event修改keys
            event2index=node2index(event)#处理索引
            x_feat, tree, rootfeat, rootindex = constructMat(event2index)
            #x_x = getfeature(x_word, x_index) # *5000
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_feat), np.array(rootindex), np.array(y)
            np.savez(os.path.join(Path,'data/Bitcoingraph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            return None
        #x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
        x_feat, tree, rootfeat, rootindex = constructMat(event2index)
        x_x = x_feat
        return rootfeat, tree, x_x, [rootindex]

    print("loading dataset", )
    #for eid in tqdm(event):
    #    loadEid(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) 
    results = Parallel(n_jobs=1, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    return

if __name__ == '__main__':
    main()
