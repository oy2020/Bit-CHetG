"""
3、生成最终子图结果
"""

#合并txt文件
from collections import OrderedDict
# 创建新文件
with open('bitcoin_id_label.txt', 'w') as merged_file:
    # 循环遍历每个文件
    for i in range(1,50):
        file_name="elliptic_10_"+str(i).zfill(2)+"T_id_label.txt"
        # 打开文件
        with open(file_name, 'r') as file:
            # 逐行写入新文件中
            for line in file:
                merged_file.write(line)

# 创建新文件
with open('bitcointree.txt', 'w') as merged_file:
    # 循环遍历每个文件
    for i in range(1,50):
        file_name="elliptictree_10_"+str(i).zfill(2)+"T.txt"
        # 打开文件
        with open(file_name, 'r') as file:
            # 逐行写入新文件中
            for line in file:
                merged_file.write(line)


# l1=[]
# l2=[]
# labelDic={}
# F=[]
# T=[]
# for line in open('bitcoin_id_label.txt'):
#     line = line.rstrip()
#     eid,label = line.split(' ')[0], line.split(' ')[1]
#     labelDic[eid] = int(label)
#     if labelDic[eid]==0:
#         F.append(eid)
#         l1 += 1
#         if labelDic[eid]==1:
#             T.append(eid)
#             l2 += 1
#     print(len(labelDic))
#     print(l1, l2) 
