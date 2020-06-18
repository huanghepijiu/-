import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn import ensemble
import numpy as np20141223前20天数据
# 导入第三方包
from sklearn import model_selection

# 读入数据
Load_Waether_data = pd.read_excel(r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190817\input\20141223前20天数据.xlsx',dtype=np.float32)


#预测用数据集
data_pred1=pd.read_excel(io=r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190817\input\20141223.xlsx',
                  header=0)
data_pred = data_pred1.columns[1:7]

# 取出自变量名称
X = Load_Waether_data.columns[1:7]

#空列表

y_predx=pd.DataFrame()

for i in range(7,103):
	y = Load_Waether_data.columns[i:i+1]
	# 将数据集拆分为训练集和测试集
	X_train, X_test, y_train, y_test = model_selection.train_test_split(Load_Waether_data[X],Load_Waether_data[y],test_size = 0.25, random_state = 1234)
	gbr =ensemble.GradientBoostingClassifier(n_estimators=300, max_depth=2, min_samples_split=2, learning_rate=0.1)
	
   gbr.fit(X_train*10000, y_train*10000)
	y_pred = gbr.predict(data_pred1[data_pred])
	y_predy=pd.DataFrame(y_pred/10000)
	y_predx=pd.concat([y_predx,y_predy],axis=1)




y_predx=pd.DataFrame(columns=['T0000','T0015','T0030','T0045','T0100','T0115','T0130','T0145',
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
                                   ], data = (y_predx.values))
print(y_predx)

y_predx=pd.concat([data_pred1['YMD'],y_predx],axis=1,join='inner')



y_predx.to_excel('/20141223.xlsx',index = False)#输出结果，写入文件


