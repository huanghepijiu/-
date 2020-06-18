# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:31:32 2019

@author: 黄河脾酒
"""


from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn import tree
# 导入第三方模块
import pandas as pd
# 读入数据
Load_Waether_data = pd.read_excel(r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190623\input\20141223前20天数据.xlsx')
Load_Waether_data.head()
#print(Load_Waether_data.shape)

# 取出自变量名称
X = Load_Waether_data.columns[1:7]

y = Load_Waether_data.columns[7:103]
#print(y)

#X,y = Load_Waether_data.ix[:,1:7],Load_Waether_data.ix[:,7:103]

# 导入第三方包
from sklearn import model_selection
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(Load_Waether_data[X],Load_Waether_data[y],test_size = 0.25, random_state = 1234)

# 导入第三方模块
from sklearn.model_selection import GridSearchCV
from sklearn import tree
# 预设各参数的不同选项值
max_depth = [5,6,7,8,9]
min_samples_split = [2,4,6,8]
min_samples_leaf = [2,4,8]
parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}

# 网格搜索法，测试不同的参数值
grid_dtreg = GridSearchCV(estimator = tree.DecisionTreeRegressor(), param_grid = parameters, cv=10)

# 模型拟合
grid_dtreg.fit(X_train, y_train)

# 返回最佳组合的参数值
print("================返回最佳组合的参数值====================")
print(grid_dtreg.best_params_)




# 构建用于回归的决策树
CART_Reg = tree.DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 2, min_samples_split = 4)
# 回归树拟合
CART_Reg.fit(X_train, y_train)
# 模型在测试集上的预测
pred = CART_Reg.predict(X_test)

pred1 = CART_Reg.predict(X_train)


#print(pred)
from sklearn import metrics
# 计算衡量模型好坏的MSE值
print("================计算衡量模型好坏的MSE值====================")
print(metrics.mean_absolute_error(y_test, pred))
print("================计算衡量训练集模型好坏的MSE值====================")
print(metrics.mean_absolute_error(y_train, pred1))






'''

#pred=pd.DataFrame(pred)
#pred.to_excel('D://sublime_Program//Pyhton Data analysis and mining practice//test20190623//output//data_pred(yizhou).xlsx',index = False)#输出结果，写入文件

data_pred1=pd.read_excel(io=r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190521\input\20141223.xlsx',
	               header=0)
data_pred = data_pred1.columns[1:7]
#print(data_pred)

#预测
y_pred = CART_Reg.predict(data_pred1[data_pred])

#print(y_pred)

y_pred=pd.DataFrame(y_pred)
pred_tree=pd.DataFrame(columns=['T0000','T0015','T0030','T0045','T0100','T0115','T0130','T0145',
                                 'T0200','T0215','T0230','T0245','T0300','T0315','T0330','T0345',
                                 'T0400','T0415','T0430','T0445','T0500','T0515','T0530','T0545',
                                 'T0600','T0615','T0630','T0645','T0700','T0715','T0730','T0745',
                                 'T0800','T0815','T0830','T0845','T0900','T0915','T0930','T0945',
                                 'T1000','T1015','T1030','T1045','T1100','T1115','T1130','T1145',
                                 'T1200','T1215','T1230','T1245','T1300','T1315','T1330','T1345',
                                 'T1400','T1415','T1430','T1445','T1500','T1515','T1530','T1545',
                                 'T1600','T1615','T1630','T1645','T1700','T1715','T1730','T1745',
                                 'T1800','T1815','T1830','T1845','T1900','T1915','T1930','T1945',
                                 'T2000','T2015','T2030','T2045','T2100','T2115','T2130','T2145',
                                 'T2200','T2215','T2230','T2245','T2300','T2315','T2330','T2345'
                                   ], data = (y_pred.values))
pred_tree=pd.concat([data_pred1['YMD'],pred_tree],axis=1,join='inner')
pred_tree.to_excel('D://sublime_Program//Pyhton Data analysis and mining practice//test20190623//output//决策树60天数据预测20141223.xlsx',index = False)#输出结果，写入文件


'''