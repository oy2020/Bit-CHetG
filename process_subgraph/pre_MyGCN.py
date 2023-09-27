import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected


import networkx as nx               
import matplotlib.pyplot as plt     
import gtx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
1、处理elliptic数据集
"""

# Load Dataframe
df_edge = pd.read_csv('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')\
        .rename(columns={'index':'txId1','index2':'txId2'})#[txId1,txId2]
df_class = pd.read_csv('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')\
        .rename(columns={'index':'txId'})[['txId','class']]#[txId,class]
df_features = pd.read_csv('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/elliptic_bitcoin_dataset/elliptic_txs_features.csv',header=None)#[Id,time,features]

# Setting Column name
df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
df_features.rename(columns={'index':'id'})
#edge:
print('Number of edges: {}'.format(len(df_edge)))

#node:[index,id]
all_nodes = list(set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id']))) #203769个点
nodes_df = pd.DataFrame(all_nodes,columns=['id']).reset_index() 
print('Number of nodes: {}'.format(len(nodes_df)))

# #以下index代替id
# # [txId1,txId2]:用index代替id
# df_edge = df_edge.join(nodes_df.rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='inner') \
#        .join(nodes_df.rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='inner',rsuffix='2') \
#        .drop(columns=['txId1','txId2']) \
#        .rename(columns={'index':'txId1','index2':'txId2'})
# df_edge.to_csv('elliptic_bitcoin_dataset_cont/df_edge.csv',index_label="index")
# #[txId,class]:用index代替id
# df_class = df_class.join(nodes_df.rename(columns={'id':'txId'}).set_index('txId'),on='txId',how='inner') \
#         .drop(columns=['txId']).rename(columns={'index':'txId'})[['txId','class']]
# #[id timestep features]:index代替id
# df_features = df_features.join(nodes_df.set_index('id'),on='id',how='inner') \
#         .drop(columns=['id']).rename(columns={'index':'id'})
# df_features = df_features [ ['id']+list(df_features.drop(columns=['id']).columns) ]

df_edge_time = df_edge.join(df_features[['id','time step']].rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='left',rsuffix='1') \
.join(df_features[['id','time step']].rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='left',rsuffix='2')
df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2'] #true
df_edge_time_fin = df_edge_time[['txId1','txId2','time step']].rename(columns={'txId1':'source','txId2':'target','time step':'time'})#[source,target,time]

#graph_edge:[time,source,target]
df_class['class'] =  df_class['class'].apply(lambda x: '3'  if x =='unknown' else x).astype(int)-1
graph_edge= df_edge_time[['time step','txId1','txId2']].rename(columns={'time step':'time','txId1':'source','txId2':'target'})

#graph_node:[time,node,class，feature]
graph_node=pd.merge(df_class.rename(columns={'txId':'id'}),df_features[['id','time step']],on='id',how='left')[['time step','id' ,'class']].rename(columns={'time step':'time'})
graph_node=pd.merge(graph_node,df_features.drop('time step',axis=1),on='id',how='left')
#graph:[time,source,target,label_s,label_t,feature_t]
graph=graph_edge.join(df_class.rename(columns={'txId':'source'})[['source','class']].set_index('source'),on='source',how='left').join(df_class.rename(columns={'txId':'target'}).set_index('target'),on='target',how='left',rsuffix='_t')
graph=graph.join(graph_node.rename(columns={'id':'target'}).set_index('target'),on='target',how='left',rsuffix='_t').drop(columns=['time_t'])

for i in range(49):
    graph_i=graph[graph['time']==i+1]
    graph_i.to_csv('Data_pre/graph_{}.csv'.format(i+1),index=False)
    
    graph_node_i=graph_node[graph_node['time']==i+1]
    graph_node_i.to_csv('Data_pre/node_graph_{}.csv'.format(i+1),index=False)

    graph_edge_i=graph_edge[graph_edge['time']==i+1]
    graph_edge_i.to_csv('Data_pre/edge_graph_{}.csv'.format(i+1),index=False)

