import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#定义学习函数
def logsig(x):
    return 1/(1+np.exp(-x))

#气象因素数据
Load_Waether_data=pd.read_excel(io=r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190618\input\20141223前20天数据.xlsx',header=-1)


最高温度 = Load_Waether_data['最高温度℃'].as_matrix()

最低温度 = Load_Waether_data['最低温度℃'].as_matrix()

平均温度 = Load_Waether_data['平均温度℃'].as_matrix()

相对湿度=Load_Waether_data['相对湿度(平均)'].as_matrix()

降雨量 = Load_Waether_data['降雨量（mm）'].as_matrix()

日期类型 = Load_Waether_data['日期类型'].as_matrix()

T0000=Load_Waether_data['T0000'].as_matrix()
T0015=Load_Waether_data['T0015'].as_matrix()
T0030=Load_Waether_data['T0030'].as_matrix() 
T0045=Load_Waether_data['T0045'].as_matrix() 
T0100=Load_Waether_data['T0100'].as_matrix() 
T0115=Load_Waether_data['T0115'].as_matrix() 
T0130=Load_Waether_data['T0130'].as_matrix() 
T0145=Load_Waether_data['T0145'].as_matrix() 
T0200=Load_Waether_data['T0200'].as_matrix() 
T0215=Load_Waether_data['T0215'].as_matrix() 
T0230=Load_Waether_data['T0230'].as_matrix() 
T0245=Load_Waether_data['T0245'].as_matrix() 
T0300=Load_Waether_data['T0300'].as_matrix() 
T0315=Load_Waether_data['T0315'].as_matrix() 
T0330=Load_Waether_data['T0330'].as_matrix() 
T0345=Load_Waether_data['T0345'].as_matrix() 
T0400=Load_Waether_data['T0400'].as_matrix() 
T0415=Load_Waether_data['T0415'].as_matrix() 
T0430=Load_Waether_data['T0430'].as_matrix() 
T0445=Load_Waether_data['T0445'].as_matrix() 
T0500=Load_Waether_data['T0500'].as_matrix() 
T0515=Load_Waether_data['T0515'].as_matrix() 
T0530=Load_Waether_data['T0530'].as_matrix() 
T0545=Load_Waether_data['T0545'].as_matrix() 
T0600=Load_Waether_data['T0600'].as_matrix() 
T0615=Load_Waether_data['T0615'].as_matrix() 
T0630=Load_Waether_data['T0630'].as_matrix() 
T0645=Load_Waether_data['T0645'].as_matrix()
T0700=Load_Waether_data['T0700'].as_matrix() 
T0715=Load_Waether_data['T0715'].as_matrix() 
T0730=Load_Waether_data['T0730'].as_matrix() 
T0745=Load_Waether_data['T0745'].as_matrix()
T0800=Load_Waether_data['T0800'].as_matrix()
T0815=Load_Waether_data['T0815'].as_matrix()
T0830=Load_Waether_data['T0830'].as_matrix()
T0845=Load_Waether_data['T0845'].as_matrix()
T0900=Load_Waether_data['T0900'].as_matrix()
T0915=Load_Waether_data['T0915'].as_matrix()
T0930=Load_Waether_data['T0930'].as_matrix()
T0945=Load_Waether_data['T0945'].as_matrix()
T1000=Load_Waether_data['T1000'].as_matrix()
T1015=Load_Waether_data['T1015'].as_matrix()
T1030=Load_Waether_data['T1030'].as_matrix()
T1045=Load_Waether_data['T1045'].as_matrix()
T1100=Load_Waether_data['T1100'].as_matrix()
T1115=Load_Waether_data['T1115'].as_matrix()
T1130=Load_Waether_data['T1130'].as_matrix()
T1145=Load_Waether_data['T1145'].as_matrix()
T1200=Load_Waether_data['T1200'].as_matrix()
T1215=Load_Waether_data['T1215'].as_matrix()
T1230=Load_Waether_data['T1230'].as_matrix()
T1245=Load_Waether_data['T1245'].as_matrix()
T1300=Load_Waether_data['T1300'].as_matrix()
T1315=Load_Waether_data['T1315'].as_matrix()
T1330=Load_Waether_data['T1330'].as_matrix()
T1345=Load_Waether_data['T1345'].as_matrix()
T1400=Load_Waether_data['T1400'].as_matrix()
T1415=Load_Waether_data['T1415'].as_matrix()
T1430=Load_Waether_data['T1430'].as_matrix()
T1445=Load_Waether_data['T1445'].as_matrix()
T1500=Load_Waether_data['T1500'].as_matrix()
T1515=Load_Waether_data['T1515'].as_matrix()
T1530=Load_Waether_data['T1530'].as_matrix()
T1545=Load_Waether_data['T1545'].as_matrix()
T1600=Load_Waether_data['T1600'].as_matrix()
T1615=Load_Waether_data['T1615'].as_matrix()
T1630=Load_Waether_data['T1630'].as_matrix()
T1645=Load_Waether_data['T1645'].as_matrix()
T1700=Load_Waether_data['T1700'].as_matrix()
T1715=Load_Waether_data['T1715'].as_matrix()
T1730=Load_Waether_data['T1730'].as_matrix()
T1745=Load_Waether_data['T1745'].as_matrix()
T1800=Load_Waether_data['T1800'].as_matrix()
T1815=Load_Waether_data['T1815'].as_matrix()
T1830=Load_Waether_data['T1830'].as_matrix()
T1845=Load_Waether_data['T1845'].as_matrix()
T1900=Load_Waether_data['T1900'].as_matrix()
T1915=Load_Waether_data['T1915'].as_matrix()
T1930=Load_Waether_data['T1930'].as_matrix()
T1945=Load_Waether_data['T1945'].as_matrix()
T2000=Load_Waether_data['T2000'].as_matrix()
T2015=Load_Waether_data['T2015'].as_matrix()
T2030=Load_Waether_data['T2030'].as_matrix()
T2045=Load_Waether_data['T2045'].as_matrix()
T2100=Load_Waether_data['T2100'].as_matrix()
T2115=Load_Waether_data['T2115'].as_matrix()
T2130=Load_Waether_data['T2130'].as_matrix()
T2145=Load_Waether_data['T2145'].as_matrix()
T2200=Load_Waether_data['T2200'].as_matrix()
T2215=Load_Waether_data['T2215'].as_matrix()
T2230=Load_Waether_data['T2230'].as_matrix()
T2245=Load_Waether_data['T2245'].as_matrix()
T2300=Load_Waether_data['T2300'].as_matrix()
T2315=Load_Waether_data['T2315'].as_matrix()
T2330=Load_Waether_data['T2330'].as_matrix()
T2345=Load_Waether_data['T2345'].as_matrix()


#将数据转换成矩阵，并使用最大最小归一数据

#输入
samplein = np.mat([最高温度,最低温度,平均温度,相对湿度,降雨量,日期类型]) #6行* 列
sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0],samplein.max(axis=1).T.tolist()[0]]).transpose()

#输出
sampleout = np.mat([T0000,T0015,T0030,T0045,T0100,T0115,T0130,T0145,
                    T0200,T0215,T0230,T0245,T0300,T0315,T0330,T0345,
                    T0400,T0415,T0430,T0445,T0500,T0515,T0530,T0545,
                    T0600,T0615,T0630,T0645,T0700,T0715,T0730,T0745,
                    T0800,T0815,T0830,T0845,T0900,T0915,T0930,T0945,
                    T1000,T1015,T1030,T1045,T1100,T1115,T1130,T1145,
                    T1200,T1215,T1230,T1245,T1300,T1315,T1330,T1345,
                    T1400,T1415,T1430,T1445,T1500,T1515,T1530,T1545,
                    T1600,T1615,T1630,T1645,T1700,T1715,T1730,T1745,
                    T1800,T1815,T1830,T1845,T1900,T1915,T1930,T1945,
                    T2000,T2015,T2030,T2045,T2100,T2115,T2130,T2145,
                    T2200,T2215,T2230,T2245,T2300,T2315,T2330,T2345])#2*20

sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0],sampleout.max(axis=1).T.tolist()[0]]).transpose()


sampleinnorm = (2*(np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.transpose()[1]-sampleinminmax.transpose()[0])-1).transpose()
print(sampleinnorm)
print('=========================================')
sampleoutnorm = (2*(np.array(sampleout.T).astype(float)-sampleoutminmax.transpose()[0])/(sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0])-1).transpose()
print(sampleoutnorm)

#给输入样本添加噪声
noise = 0.03*np.random.rand(sampleoutnorm.shape[0],sampleoutnorm.shape[1])
sampleoutnorm += noise



#========================================
'''

maxepochs=50000;                              %最多训练次数为50000
learnrate=0.035;                                       %学习速率为0.035
errorfinal=0.65*10^(-3);                              %目标误差为0.65*10^(-3)
InDim=3;                    %网络输入维度为3
OutDim=2;                   %网络输出维度为2
SamNum=20;                  %输入样本数量为20
TestSamNum=20;              %测试样本数量也是20
ForcastSamNum=2;            %预测样本数量为2
HiddenUnitNum=8;            %中间层隐节点数量取8,比工具箱程序多了1个
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;   %初始化输入层与隐含层之间的权值
B1=0.5*rand(HiddenUnitNum,1)-0.1;       %初始化输入层与隐含层之间的阈值
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1; %初始化输出层与隐含层之间的权值              
B2=0.5*rand(OutDim,1)-0.1;                %初始化输出层与隐含层之间的阈值

'''
#========================================



#定义模型的参数
 #最多训练次数
maxepochs = 100000
 #学习速率为0.035
learnrate = 0.005
#目标误差为0.65*10^(-3)
errorfinal = 0.65*10**(-3)


###############################
#
#参数调节
#
###############################
#输入样本数量为20！！！！！！！！！！！！！！！！
samnum = 20
#网络输入维度为3
indim = 6
#网络输出维度为2
outdim = 96
#中间层隐节点数量
hiddenunitnum = 30


w1 = 0.5*np.random.rand(hiddenunitnum,indim)-0.1
b1 = 0.5*np.random.rand(hiddenunitnum,1)-0.1
w2 = 0.5*np.random.rand(outdim,hiddenunitnum)-0.1
b2 = 0.5*np.random.rand(outdim,1)-0.1



#给中间变量预先占据内存
errhistory = []

#开始训练模型
for i in range(maxepochs):
    #隐含层网络输出
    hiddenout = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
    # 输出层网络输出
    networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
    #实际输出与网络输出之差
    err = sampleoutnorm - networkout

    #能量函数（误差平方和）
    sse = sum(sum(err**2))
    errhistory.append(sse)
    
    #如果达到误差要求则跳出学习循环
    if sse < errorfinal:
        break
    # 以下六行是BP网络最核心的程序
    # 他们是权值（阈值）依据能量函数负梯度下降原理所作的每一步动态调整量

    delta2 = err
    delta1 = np.dot(w2.transpose(),delta2)*hiddenout*(1-hiddenout)
    dw2 = np.dot(delta2,hiddenout.transpose())
    db2 = np.dot(delta2,np.ones((samnum,1)))
    dw1 = np.dot(delta1,sampleinnorm.transpose())
    db1 = np.dot(delta1,np.ones((samnum,1)))

    #对输出层与隐含层之间的权值和阈值进行修正
    w2 += learnrate*dw2
    b2 += learnrate*db2

    #对输入层与隐含层之间的权值和阈值进行修正
    w1 += learnrate*dw1
    b1 += learnrate*db1


#绘制误差曲线图
errhistory10 = np.log10(errhistory)
minerr = min(errhistory10)
plt.plot(errhistory10)
plt.plot(range(0,i+1000,1000),[minerr]*len(range(0,i+1000,1000)))
ax = plt.gca()
ax.set_yticks([-2,-1,0,1,minerr,2])
ax.set_yticklabels([u'$10^{-2}$',u'$10^{-1}$',u'$10^{1}$',u'$10^{2}$',str(('%.4f'%np.power(10,minerr)))])
ax.set_xlabel('iteration')
ax.set_ylabel('error')
ax.set_title('Error Histroy')

plt.savefig('errorhistory30.png',dpi=700)
plt.close() 




'''

#实现方针输出和实际输出对比图

#隐含层输出最终结果
hiddenout = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
#输出层输出最终结果
networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()


print('====================networkout=====================')
print(networkout)


#还原网络输出层的结果
diff = sampleoutminmax[:,1]-sampleoutminmax[:,0]
networkout2 = (networkout+1)/2

for i in range (0,96):
    if i<=96:
        networkout2[i] = networkout2[i]*diff[i]+sampleoutminmax[i][0]


#矩阵转置
networkout2=networkout2.T
networkout2 = pd.DataFrame(networkout2)
print('====================networkout2=====================')
print(networkout2)

outputfile='D://sublime_Program//Pyhton Data analysis and mining practice//test20190618//output//networkout2.xlsx'
networkout2.to_excel(outputfile,index = False)#输出结果，写入文件

'''


#预测
'''
% 利用训练好的网络进行预测
% 当用训练好的网络对新数据pnew进行预测时，也应作相应的处理
pnew=[73.39 75.55
      3.9635 4.0975
      0.9880 1.0268];                     %2010年和2011年的相关数据；
pnewn=tramnmx(pnew,minp,maxp);         %利用原始输入数据的归一化参数对新数据进行归一化；
HiddenOut=logsig(W1*pnewn+repmat(B1,1,ForcastSamNum)); % 隐含层输出预测结果
anewn=W2*HiddenOut+repmat(B2,1,ForcastSamNum);           % 输出层输出预测结果

%把网络预测得到的数据还原为原始的数量级；
anew=postmnmx(anewn,mint,maxt)

'''


'''
pnew=pd.read_excel(io=r'D:\sublime_Program\Pyhton Data analysis and mining practice\test20190618\input\预测数据集7天.xlsx',header=0)



最高温度预测 = pnew['最高温度℃'].as_matrix()

最低温度预测 = pnew['最低温度℃'].as_matrix()

平均温度预测 = pnew['平均温度℃'].as_matrix()

相对湿度预测=pnew['相对湿度(平均)'].as_matrix()

降雨量预测 = pnew['降雨量（mm）'].as_matrix()

日期类型预测 = pnew['日期类型'].as_matrix()



#输入
pnew = np.mat([最高温度预测,最低温度预测,平均温度预测,相对湿度预测,降雨量预测,日期类型预测]) #6行* 列
pnewminmax = np.array([pnew.min(axis=1).T.tolist()[0],pnew.max(axis=1).T.tolist()[0]]).transpose()
pnewnorm = (2*(np.array(pnew.T)-pnewminmax.transpose()[0])/(pnewminmax.transpose()[1]-pnewminmax.transpose()[0])-1).transpose()

hiddenout = logsig((np.dot(w1,pnewnorm).transpose()+b1.transpose())).transpose()
#输出层输出最终结果
networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()


#还原网络输出层的预测结果
diff = sampleoutminmax[:,1]-sampleoutminmax[:,0]
networkout2 = (networkout+1)/2

for i in range (0,96):
    if i<=96:
        networkout2[i] = networkout2[i]*diff[i]+sampleoutminmax[i][0]






#矩阵转置
networkout2=networkout2.T
#矩阵转dataframe
#networkout2 = pd.DataFrame(networkout2)
print('====================networkout2=====================')
print(networkout2)
#给预测结果加上表头

result_reg=pd.DataFrame(columns=['T0000','T0015','T0030','T0045','T0100','T0115','T0130','T0145',
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
                                   ], data = (networkout2))

print(result_reg)
result_reg.to_excel('D://sublime_Program//Pyhton Data analysis and mining practice//test20190618//output//前20天数据预测神经网络预测20141223.xlsx',index = False)#输出结果，写入文件

'''



