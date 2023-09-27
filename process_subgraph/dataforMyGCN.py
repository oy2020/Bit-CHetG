import networkx as nx               #载入networkx包
import matplotlib.pyplot as plt     #用于画图
import pandas as pd
import numpy as np
import time
import copy
from collections import deque
# import graphviz

'''
2、从全图中产生子图
规则：
广度优先搜索N个节点,标签是非法节点的个数
'''
# graph

import pandas as pd

class Node:
    def __init__(self,num,label,features=None):
        self.num = num
        self.label=label
        self.features = features
        self.children = []

class SubTree:
    def __init__(self, root_node):
        self.root = root_node
        
    def add_child(self, parent_node, child_node):
        parent_node.children.append(child_node)

def build_subgraph(df,start_node_num,n,count_illicit):
    """
    df:[time source target class class_t]
    start_node_num
    n:邻居节点个数
    count_illicit:非法节点个数
    """
    start_node = df.loc[df['source'] == start_node_num]
    if len(start_node) == 0:
        raise ValueError("Start node not found in dataframe")
    if(list(start_node['class'])[0]==0):
        count=[1]
    else:
        count=[0]
    start_node = Node(start_node_num,list(start_node['class'])[0])
    subgraph = SubTree(start_node)

    # 创建一个队列，并将起始节点入队
    q = deque([start_node])
    visited = set([start_node_num])
    depth = 0

    while q and depth < n-1:
        # 取出队列中的节点
        curr_node = q.popleft()

        # 查找当前节点的邻居节点
        neighbors = df.loc[df['source'] == curr_node.num]

        # 添加邻居节点到子树中
        for _, neighbor in neighbors.iterrows():
            neighbor_num = neighbor['target']
            if neighbor_num not in visited and depth < n-1:
                visited.add(neighbor_num)
                neighbor_node = Node(neighbor_num, neighbor['class_t'])
                subgraph.add_child(curr_node, neighbor_node)
                if neighbor['class_t'] == 0:
                    count[0] += 1
                q.append(neighbor_node)
                depth+=1
        

    count_illicit.append(count[0])
    return subgraph

    # build_subgraph_helper(df, start_node, subgraph, n,1,count)
    # count_illicit.append(count[0])
    # return subgraph

# def build_subgraph_helper(df, parent_node, subgraph, n, depth,count):
#     children = df.loc[df['source'] == parent_node.num]
#     #将所有叶节点加进去
#     for _, child in children.iterrows():
#         if depth<n:
#             child_node = Node(child['target'],child['class_t'])
#             subgraph.add_child(parent_node, child_node)
#             if child['class_t']==0:
#                 count[0]=count[0]+1
#             depth=depth+1
#         else:
#             return
#     #depth < n:将child_noded的叶节点加进去
#     for _, child in children.iterrows():
#         child_node = Node(child['target'],child['class_t']) 
#         build_subgraph_helper(df, child_node, subgraph, n, depth,count)
    
            
        

def store_subtree(node,subtreeList):#node=subTree.root
    for child in node.children:
        subtreeList.append([node.num,child.num])
    for child in node.children:
        store_subtree(child,subtreeList)

def draw_subtree(subtreeList,i,N,graph,rootlabel):
    index=str(int(subtreeList[1][0])).zfill(9)
    G = nx.DiGraph()
    x=subtreeList[1:]
     #添加节点标签：
    node=set(np.array(x).flatten())
    nodes=[]
    for n in node:
        if len(graph[graph['target']==n])==0:
            label=str(list(graph[graph['source']==n]['class'])[0])
        else:
            label=str(list(graph[graph['target']==n]['class_t'])[0])
        nodes.append((n,{'label':label}))
    G.add_nodes_from(nodes)
    G.add_edges_from(x)
    #添加颜色：
    label_colors= {'0': 'red', '1': 'blue', '2': 'green'}
    labels=nx.get_node_attributes(G, 'label')
    #绘图：
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos, labels=nx.get_node_attributes(G, 'label'), node_color=[label_colors[label] for label in labels.values()],with_labels=True)
    plt.axis('off')
    plt.savefig("Data_subgraph_analyse/root{}_时刻{}节点{}的{}个邻居.png".format(rootlabel,time,index,N))
    plt.close()           
    # #plt.show()

if __name__ == '__main__':
    T1 = time.time()
    N=10
    M=1

    for i in range(49):
        print(i)
        time=str(i+1).zfill(2)
        edge=pd.read_csv('Data_pre/edge_graph_{}.csv'.format(i+1)) #[time source target]
        node=pd.read_csv('Data_pre/node_graph_{}.csv'.format(i+1)) #[time id fetures]
        graph=pd.read_csv('Data_pre/graph_{}.csv'.format(i+1)) #[time source target class fetures]
        
        storeLabel=[] #[num,label]
        storeTree=[] #[num,source,target,feature]
        count_illicit_illicitnode=[]
        count_illicit_edge=[]
        count_licit_illicitnode=[]
        count_licit_edge=[]
        illicit_root_list=[]
        licit_root_list=[]
        tag=[]
        txt1=[]
        txt2=[]
        
        for row in graph.iterrows():
            if(row[1][3]==0):
                startNode=row[1][1] #source
                rootnum=int(startNode)
                if(rootnum not in tag):
                    #txt1
                    id=str(i+1).zfill(2)+str(int(startNode)).zfill(9)
                    label=0

                    subtree=build_subgraph(graph, startNode, N,count_illicit_illicitnode)
                    subtreeList=[['None',subtree.root.num]]
                    store_subtree(subtree.root,subtreeList)
                    count_illicit_edge.append(len(subtreeList))
                    illicit_root_list.append(rootnum)
                    draw_subtree(subtreeList,i,N,graph,label)
                    illicitInfo=[illicit_root_list,count_illicit_edge,count_illicit_illicitnode]

                    if len(subtreeList)>5:
                        c=count_illicit_illicitnode[-1]/len(subtreeList)
                        txt1.append([id,c])
                        #txt2
                        for j in subtreeList:
                            target_feature=node[node['id']==j[1]].iloc[:,2:168].values.tolist()[0]
                            if j[0]!='None':
                                txt2.append([id,int(j[0]),int(j[1])]+target_feature)
                            elif(j[0]=='None'):
                                txt2.append([id,j[0],int(j[1])]+target_feature)
                    tag.append(rootnum)
                # print(illicitInfo)
                #print(subtreeList)

            elif(row[1][3]==1):
                startNode=row[1][1] #source
                rootnum=int(startNode)
                if(rootnum not in tag):
                    #txt1
                    id=str(i+1).zfill(2)+str(int(startNode)).zfill(9)
                    label=1
                    # txt1.append([id,label])
                    subtree=build_subgraph(graph, startNode, N,count_licit_illicitnode)
                    subtreeList=[['None',subtree.root.num]]
                    store_subtree(subtree.root,subtreeList)

                    count_licit_edge.append(len(subtreeList))
                    # draw_subtree(subtreeList,i,N,graph,label)
                    licit_root_list.append(rootnum)
                    licitInfo=[licit_root_list,count_licit_edge,count_licit_illicitnode]
                    
                    if len(subtreeList)>5:
                        c=count_licit_illicitnode[-1]/len(subtreeList)
                        txt1.append([id,c])
                    #txt2
                        for j in subtreeList:
                            target_feature=node[node['id']==j[1]].iloc[:,2:168].values.tolist()[0]
                            if j[0]!='None':
                                txt2.append([id,int(j[0]),int(j[1])]+target_feature)
                            elif(j[0]=='None'):
                                txt2.append([id,j[0],int(j[1])]+target_feature)
                    tag.append(rootnum)


        store_info_illicit=pd.DataFrame(illicitInfo)
        store_info_illicit.to_csv('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/Data_subgraph_analyse/illicitInfo_{}.csv'.format(i), header=False, index=False)
        store_info_licit=pd.DataFrame(licitInfo)
        store_info_licit.to_csv('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/Data_subgraph_analyse/licitInfo_{}.csv'.format(i), header=False, index=False)
        
        with open('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/DataforBiGCN_subgraph/elliptic_{}_{}T_id_label.txt'.format(N,time),'w') as f:
            for i in txt1:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

        with open('/home/catlab/oysy/bitcoin_graph/MyGcnGraph/DataforBiGCN_subgraph/elliptictree_{}_{}T.txt'.format(N,time),'w') as f:
            for i in txt2:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()
    
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
