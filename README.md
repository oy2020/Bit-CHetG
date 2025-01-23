# Bit-CHetG
Code and dataset for Bit-CHetG    
****  
1、requirement：  
torch==1.4.0   
torch_scatter==1.4.0  
torch_sparse==0.4.3  
torch_cluster==1.4.5  
torch_geometric==1.3.2  
tqdm==4.40.0  
joblib==0.14.1  
****  
2、process subgraph  
python /process_subgraph/pre_MyGCN.py  
python /process_subgraph/dataforMyGCN.py  
python /process_subgraph//DataforBiGCN_subgraph/ReforBiGCN.py  
****  
3、Bit-CHetG model  
#无数据增强  
python Bit-CHetG-master/Process/getBitcoingraph.py  
#二元交叉熵  
python Bit-CHetG-master/model/Bitcoin/BIGCN_Bitcoin.py 100  
#交叉熵+对比损失  
python Bit-CHetG-master/model/Bitcoin/BIGCN_Bitcoin_CL.py 100  

****
dataset can be downloaded in baiduyunpan：

BlockSec:
链接: https://pan.baidu.com/s/1wNktynHDNSou6dKTPzwu4A?pwd=ikya 提取码: ikya   

Elliptic:
链接: https://pan.baidu.com/s/1cdNNwMMSliVL2F_a_OqwAQ?pwd=y392 提取码: y392
