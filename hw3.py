# -*- coding: utf-8 -*-
import math
import numpy as np
from sklearn import datasets


def entropy(p1,n1): #p:positive/n:negative
    if(p1==0 and n1==0):
        return 1
    elif(p1==0):
        return 0
    elif(n1==0):
        return 0 
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    return -pp*math.log2(pp)-pn*math.log2(pn)

def infogain(p1,n1,p2,n2):
    num1 = p1+n1
    num2 = p2+n2
    num = num1+num2
    return entropy(p1+p2,n1+n2)-(num1/num*entropy(p1,n1)+num2/num*entropy(p2,n2))
#挑infomation gain最佳的分法來產生tree分支

iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
feature = iris.data[idx,:]
target = iris.target[idx]
pred = []

def tree(train_f,train_t,test_f,test_t):
    train_f01 = []
    train_t01 = []
    train_f02 = []
    train_t02 = []
    train_f12 = []
    train_t12 = []
    for i in range(len(train_t)): #把建樹的訓練資料分開
        if train_t[i]==0: #target是0的會用來建01和02的樹
            train_f01.append(train_f[i])
            train_t01.append(train_t[i])
            train_f02.append(train_f[i])
            train_t02.append(train_t[i])
        elif train_t[i]==1: #target是1的會用來建01和12的樹
            train_f01.append(train_f[i])
            train_t01.append(train_t[i])
            train_f12.append(train_f[i])
            train_t12.append(train_t[i])
        else: #target是2的會用來建02和12的樹
            train_f02.append(train_f[i])
            train_t02.append(train_t[i])
            train_f12.append(train_f[i])
            train_t12.append(train_t[i])
    train_f01 = np.array(train_f01)
    train_t01 = np.array(train_t01)
    train_f02 = np.array(train_f02)
    train_t02 = np.array(train_t02)
    train_f12 = np.array(train_f12)
    train_t12 = np.array(train_t12)
    node = dict()
    node['data'] = range(len(train_t01))
    Tree1 = [] #0-1 tree
    Tree1.append(node)
    t = 0
    while(t<len(Tree1)):
        index = Tree1[t]['data']
        if ((train_t01[index]==0).all()==True): #全部分至0,代表他是葉節點
            Tree1[t]['leaf']=1
            Tree1[t]['decision']=0 #決策樹告訴你答案是0
        elif((train_t01[index]==1).all()==True): #全部分至1,代表他是葉節點
            Tree1[t]['leaf']=1
            Tree1[t]['decision']=1 #決策樹告訴你答案是1
        else:
            bestIG = 0
            for i in range(train_f01.shape[1]):
                pool = list(set(train_f01[index,i])) #set會把重複的拿掉
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = [] #group1
                    G2 = []
                    for k in index:
                        if(train_f01[k,i]<thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = infogain(sum(train_t01[G1]==1),sum(train_t01[G1]==0),sum(train_t01[G2]==1),sum(train_t01[G2]==0))
                    if(thisIG > bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 =  G2
                        bestthres = thres 
                        bestf= i
            if(bestIG>0):
                Tree1[t]['leaf']=0
                Tree1[t]['selectf']=bestf
                Tree1[t]['threshold']=bestthres
                Tree1[t]['child']=[len(Tree1),len(Tree1)+1] #生兩個子節點在list最後面
                node = dict()
                node['data'] = bestG1
                Tree1.append(node)
                node = dict()
                node['data'] = bestG2
                Tree1.append(node)
            else:
                Tree1[t]['leaf']=1
                if(sum(train_t01[index]==1)>sum(train_t01[index]==0)):
                    Tree1[t]['decision']=1
                else:
                    Tree1[t]['decision']=0
        t+=1
    node2 = dict()
    node2['data'] = range(len(train_t02))
    Tree2 = [] #0-2 tree
    Tree2.append(node2)
    s = 0
    while(s<len(Tree2)):
        index = Tree2[s]['data']
        if ((train_t02[index]==0).all()==True): #全部分至0,代表他是葉節點
            Tree2[s]['leaf']=1
            Tree2[s]['decision']=0 #決策樹告訴你答案是0
        elif((train_t02[index]==2).all()==True): #全部分至2,代表他是葉節點
            Tree2[s]['leaf']=1
            Tree2[s]['decision']=2 #決策樹告訴你答案是2
        else:
            bestIG = 0
            for i in range(train_f02.shape[1]):
                pool = list(set(train_f02[index,i])) 
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = [] 
                    G2 = []
                    for k in index:
                        if(train_f02[k,i]<thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = infogain(sum(train_t02[G1]==2),sum(train_t02[G1]==0),sum(train_t02[G2]==2),sum(train_t02[G2]==0))
                    if(thisIG > bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 =  G2
                        bestthres = thres 
                        bestf= i
            if(bestIG>0):
                Tree2[s]['leaf']=0
                Tree2[s]['selectf']=bestf
                Tree2[s]['threshold']=bestthres
                Tree2[s]['child']=[len(Tree2),len(Tree2)+1] 
                node2 = dict()
                node2['data'] = bestG1
                Tree2.append(node2)
                node2 = dict()
                node2['data'] = bestG2
                Tree2.append(node2)
            else:
                Tree2[s]['leaf']=1
                if(sum(train_t02[index]==2)>sum(train_t02[index]==0)):
                    Tree2[s]['decision']=2
                else:
                    Tree2[s]['decision']=0
        s+=1
    node3 = dict()
    node3['data'] = range(len(train_t12))
    Tree3 = [] #1-2 tree
    Tree3.append(node3)
    v = 0
    while(v<len(Tree3)):
        index = Tree3[v]['data']
        if ((train_t12[index]==1).all()==True): #全部分至1,代表他是葉節點
            Tree3[v]['leaf']=1
            Tree3[v]['decision']=1 #決策樹告訴你答案是1
        elif((train_t12[index]==2).all()==True): #全部分至2,代表他是葉節點
            Tree3[v]['leaf']=1
            Tree3[v]['decision']=2 #決策樹告訴你答案是2
        else:
            bestIG = 0
            for i in range(train_f12.shape[1]):
                pool = list(set(train_f12[index,i])) 
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = [] 
                    G2 = []
                    for k in index:
                        if(train_f12[k,i]<thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = infogain(sum(train_t12[G1]==1),sum(train_t12[G1]==2),sum(train_t12[G2]==1),sum(train_t12[G2]==2))
                    if(thisIG > bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 =  G2
                        bestthres = thres 
                        bestf= i
            if(bestIG>0):
                Tree3[v]['leaf']=0
                Tree3[v]['selectf']=bestf
                Tree3[v]['threshold']=bestthres
                Tree3[v]['child']=[len(Tree3),len(Tree3)+1] 
                node3 = dict()
                node3['data'] = bestG1
                Tree3.append(node3)
                node3 = dict()
                node3['data'] = bestG2
                Tree3.append(node3)
            else:
                Tree3[v]['leaf']=1
                if(sum(train_t12[index]==2)>sum(train_t12[index]==1)):
                    Tree3[v]['decision']=2
                else:
                    Tree3[v]['decision']=1
        v+=1
    for i in range(len(test_t)):
        test_feature = test_f[i,:]
        now1 = 0 #現在走到哪
        while(Tree1[now1]['leaf']==0):
            if(test_feature[Tree1[now1]['selectf']]<=Tree1[now1]['threshold']):
                now1 = Tree1[now1]['child'][0]
            else:
                now1 = Tree1[now1]['child'][1]
        now2 = 0 
        while(Tree2[now2]['leaf']==0):
            if(test_feature[Tree2[now2]['selectf']]<=Tree2[now2]['threshold']):
                now2 = Tree2[now2]['child'][0]
            else:
                now2 = Tree2[now2]['child'][1]
        now3 = 0 
        while(Tree3[now3]['leaf']==0):
            if(test_feature[Tree3[now3]['selectf']]<=Tree3[now3]['threshold']):
                now3 = Tree3[now3]['child'][0]
            else:
                now3 = Tree3[now3]['child'][1]
        d0 = 0
        d1 = 0
        d2 = 0 #紀錄判斷結果為0,1,2的個數
        if (Tree1[now1]['decision']==0):d0+=1
        elif(Tree1[now1]['decision']==1):d1+=1
        else:d2+=1
        if (Tree2[now2]['decision']==0):d0+=1
        elif(Tree2[now2]['decision']==1):d1+=1
        else:d2+=1
        if (Tree3[now3]['decision']==0):d0+=1
        elif(Tree3[now3]['decision']==1):d1+=1
        else:d2+=1
        if((d0>d1)&(d0>d2)):d=0 #多數決
        elif((d1>d0)&(d1>d2)):d=1
        else:d=2
        pred.append(d)
        #print('  ',test_t[i],'    ',Tree1[now1]['decision'],'    ',Tree2[now2]['decision'],'    ',Tree3[now3]['decision'],'    ',d)
#print('target','tree01','tree02','tree12','predict')
# ------ 5 fold ------
tree(feature[30:],target[30:],feature[0:30],target[0:30])
tree(np.r_[feature[0:30],feature[60:150]],np.r_[target[0:30],target[60:150]],feature[30:60],target[30:60])  
tree(np.r_[feature[0:60],feature[90:150]],np.r_[target[0:60],target[90:150]],feature[60:90],target[60:90]) 
tree(np.r_[feature[0:90],feature[120:150]],np.r_[target[0:90],target[120:150]],feature[90:120],target[90:120])  
tree(feature[0:120],target[0:120],feature[120:150],target[120:150]) 
pred = np.array(pred)   
count = 0
confusion = np.zeros(shape=(3,3))
for j in range(len(target)):
    if (target[j]==pred[j]): count+=1
    confusion[target[j]][pred[j]]+=1
accuracy = count/150
print('accuracy =',accuracy)
print('confusion matrix')
print('    0','  1','  2')
print('0',confusion[0])
print('1',confusion[1])
print('2',confusion[2])
