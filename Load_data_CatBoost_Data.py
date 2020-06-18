# -*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd
import numpy as np

# 读取数据文件
#data = pd.read_excel(r'C:\Users\黄河脾酒\Desktop\20141223前20天数据1.xlsx')

#data1 = pd.read_excel(r'C:\Users\黄河脾酒\Desktop\20141223.xlsx')


data = pd.read_excel(r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190809\input\20141223前40天数据 - 副本.xlsx')

data1 = pd.read_excel(r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190809\input\20141223 - 副本.xlsx')



'''第一部分：缺失值的处理'''
#  因为Pm2.5是目标数据，如有缺失值直接删除这一条记录

# 删除目标值为空值的行, 其他列为缺失值则自动填充,并将目标变量放置在数据集最后一列
def DeleteTargetNan(exdata, targetstr):
    '''
    :param exdata: dataframe数据集
    :param targetstr: 目标字段名称
    :return: 预处理后的dataframe格式的数据集
    '''
    #  首先判断目标字段是否有缺失值
    if exdata[targetstr].isnull().any():
        #  首先确定缺失值的行数
        loc = exdata[targetstr][data[targetstr].isnull().values == True].index.tolist()
        #  然后删除这些行
        exdata = exdata.drop(loc)
    # 凡是有缺失值的再一起利用此行的均值填充
    exdata = exdata.fillna(exdata.mean())
    # 将目标字段至放在最后的一列
    targetnum = exdata[targetstr].copy()
    del exdata[targetstr]
    exdata[targetstr] = targetnum
    return exdata

# 删除原始数据中不需要的字段名
def Shanchu(exdata, aiduan=['YMD']):
    for ai in aiduan:
        if ai in exdata.keys():
            del exdata[ai]
    return exdata


#  因为CatBoost支持类别型特征，所以不需要进行任何的编码处理，但是需要提前声明哪些个特征是类别型特征

# 数据处理后最终的数据集
first = DeleteTargetNan(data, 'T0000')
# 去除字段后
two = Shanchu(first)

first1 = data1
# 去除字段后
two1 = Shanchu(first1)

# 将数据集按照8:2的比例分为训练、预测数据集。
# 为了便于确定最优的参数，在这里把随机的种子固定下来，也就是，作为训练、验证、预测的数据集是固定的

def fenge(exdata, per=[0.8, 0.2]):
    '''
    :param exdata: 总的数据集
    :param per: 训练、验证数据所占比例
    :return: 存储训练，验证，预测数据字典
    '''
    # 总长度
    lent = len(exdata)
    
    alist = np.arange(lent)
    np.random.seed(7)
    np.random.shuffle(alist)

    # 训练
    xunlian_length = int(lent * per[0])
    np.random.seed(13)
    xunlian = np.random.choice(alist, xunlian_length, replace=False)

    # 剩下的
    shengxai_length = np.array([i for i in alist if i not in xunlian])
    # 验证
    yanzheng_length = int(lent * per[1])
    np.random.seed(17)
    yanzheng = np.random.choice(shengxai_length, yanzheng_length, replace=False)
    # 预测
    #yuce = np.array([i for i in alist if i not in xunlian and i not in yanzheng])
    #print(yuce)
    # 存储字典
    dataic = {}

    dataic['train'] = exdata[xunlian]
    
    dataic['test'] = exdata[yanzheng]

    dataic['predict'] = two1.values
    return dataic


data_dict = fenge(two.values)

