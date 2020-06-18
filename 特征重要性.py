# !/usr/bin/env python3.7
# -*-encoding: utf-8 -*-
 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
 
df_train=pd.read_excel(io=r'C:\Users\黄河脾酒\Desktop\20141223前20天数据.xlsx',header=-1)


#df_train = pd.read_excel(r'C:\\Users\\sz009\Desktop\\tianchuang_test_20191012.xlsx', header=0, skiprows=0)
df_train.info()  # 显示数据信息
print(df_train.head(5))   # 查看前5行数据
print(df_train.tail(5))   # 查看最后5行
print(df_train)
 
t = np.around(df_train.corr(), decimals=4)  # 这里是将矩阵结果保留4位小数
tt = df_train.corr()     # 默认保留6位小数，corr = df_train.corr(method='pearson')方法选择person相关性，'spearman'秩相关
# ttt = np.corrcoef(df_train)    # 没有表头，WTF!
print(t)
print(tt)


# print(ttt)
'''
线性相关：主要采用皮尔逊pearson相关系数来度量连续变量之间的线性相关强度；
线性相关系数|r    相关程度
0<=|r|<0.3       低度相关
0.3<=|r|<0.8     中度相关
0.8<=|r|<1       高度相关
'''

'''
mm = df_train['最高温度℃'].corr(df_train['最低温度℃'])  # 进行两列之间的相关性分析
print(mm)

					

oo = df_train[['最高温度℃', '最低温度℃', '平均温度℃', '相对湿度(平均)', '降雨量（mm）']].corr()   # 计算指定多列相关系数
print(oo)
'''

plt.subplots(figsize=(6, 6))    # 设置画面大小
plt.rcParams['font.sans-serif'] = ['times new Roman']   # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题,负号正常显示
#plt.rcParams['font.sans-serif'] = ['STSong']

#plt.title('变量相关系数 - 热图\n', fontsize=18)   # 添加图表标题“变量相关系数 - 热图”,fontsize=18 字体大小 可省略
# annot=True，是显式热力图上的数值；vmax是显示最大值;xticklabels、yticklabels轴标签显示；square=True，将图变成一个正方形，默认是一个矩形；cmap="Blues"是一种模式，就是图颜色配置。
# mask:控制某个矩阵块是否显示出来,默认值是None,如果是布尔型的DataFrame，则将DataFrame里True的位置用白色覆盖掉

'''
plt.title('变量相关系数 - 热图\n', fontsize=18)
sns.heatmap(t, annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu",
            linewidths=0.05, linecolor='white', mask=t < 0.8)    # mask=t < 0.8等价于mask=(t < 0.8)
plt.show()
'''

#plt.title('变量相关系数 - 热图\n', fontsize=18)
sns.heatmap(t, annot=True,vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu",
            linewidths=0.05, linecolor='white', mask=None)
plt.savefig('./BtateRelation.pdf')
plt.show()


import os
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt

