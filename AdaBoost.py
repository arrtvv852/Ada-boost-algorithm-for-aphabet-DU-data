import os
import numpy as np
import pandas as pd



#os.chdir("C:\\Users\\cimlab\\Desktop\\SMO_kernel")
trainset = pd.read_csv("alphabet_DU_training_header.csv")
testset = pd.read_csv("alphabet_DU_testing_header.csv")



#data preprocessing
for i in range(0,len(trainset.index)):
    if trainset["class"][i]==4:
        trainset.set_value(i,"class",1)
    elif trainset["class"][i]==21:
        trainset.set_value(i,"class",-1)

for i in range(0,len(testset.index)):
    if testset["class"][i]==4:
        testset.set_value(i,"class",1)
    elif testset["class"][i]==21:
        testset.set_value(i,"class",-1)
    


#Cross Validation
############
def cross_T(n,train):

    m = len(train.index)



    size = round(m/n)

    fold=[]

    all = list(range(0,m))
    fold = []
    f = []
    for i in range(0,n-1):
        f = []
        for j in range(0,size):
            temp = all.pop(int(np.random.randint(0,len(all))))
            f.append(int(temp))
        fold.append(f)

    fold.append(all)


    low = 1.
    up = 1001.

    Tbest = 0.
    r_matrix = [0]*11
    risk = 0.
    for i in range(0,3):
        cut = 10**(2-i)
        T_range = np.array(np.linspace(low,up,11))
        count = 0
        for j in T_range:
            j = int(j)
            print("Cross Validating T, Please keep waiting. . .")
            print("T now:",j)
        
            for k in range(0,n):
                all=list(range(0,m))
                for l in np.array(fold[k]):
                    all.remove(l)
                ris,ac,out,H = AdaBoost(int(j),train.iloc[all,:],train.iloc[fold[k],:])
                risk = risk+ris
            risk = risk/n
            r_matrix[count] = risk
            count += 1
        amin = np.argmin(r_matrix)
        print(r_matrix)
        Tbest = T_range[amin]
        low = low + cut*(amin-0.5)
        up = low + cut

    return round(Tbest)
###############



##########################


#Find best base classifier
def base_class(m,n,c,base,x):
    TFcube = []
    htemp = np.repeat(0,m)
    for i in range(0,n):
        split = len(base[i])
        split += 1
        TFmatrix = np.zeros((m,2*split))
        for j in range(0,split):
            for k in 0,1:
                for l in range(0,m):
                    if k == 0:
                        if j<split-1:
                            if x[l,i]<=base[i][j]:
                                htemp[l] = -1
                            else:
                                htemp[l] = 1
                        else:
                            htemp[l] = -1
                    else:
                        if j<split-1:
                            if x[l,i]<=base[i][j]:
                                htemp[l] = 1
                            else:
                                htemp[l] = -1
                        else:
                            htemp[l] = 1
                    if htemp[l] != c[l]:
                        if k == 0:
                            TFmatrix[l,j] = 1
                        else:
                            TFmatrix[l,j+split] = 1
        TFcube.append(TFmatrix)
    return TFcube

def best_base(iterN,D,m,n,h,c,base,x,T,ht,TFcube):
    risk = np.zeros((n,3))
    for i in range(0,n):
        split = len(TFcube[i][0])
        risk_temp = np.repeat(0.,split)
        TFmatrix = np.matrix(D[iterN,:])*np.matrix(TFcube[i])
        for j in range(0,split):
            risk_temp[j] = sum(TFmatrix[:,j])
        temp = np.argmin(risk_temp)
        risk[i,0] = np.argmin(risk_temp)
        risk[i,1] = min(risk_temp)
        if risk[i,0]<split/2:
            risk[i,2] = 0
        else:
            risk[i,2] = 1
            risk[i,0] -=split/2
    temp = np.argmin(risk[:,1])
    ht[iterN,range(0,3)] = risk[temp,:]
    ht[iterN,3] = temp
    for i in range(0,m):
        if risk[temp,2] == 0:
            if risk[temp,0]<len(TFcube[temp][0])/2-1:
                if x[i,temp] <= base[temp][int(risk[temp,0])]:
                    h[iterN,i] = -1
                else:
                    h[iterN,i] = 1
            else:
                h[iterN,i] = -1
        else:
            if risk[temp,0]<len(TFcube[temp][0])/2-1:
                if x[i,temp] <= base[temp][int(risk[temp,0])]:
                    h[iterN,i] = 1
                else:
                    h[iterN,i] = -1
            else:
                h[iterN,i] = 1
    return ht,h

def update(D,m,n,h,c,base,x,T,ht,TFcube):
    alpha = np.repeat(0.,T)
    Z = np.repeat(0.,T)
    for t in range(0,T):
        ht,h = best_base(t,D,m,n,h,c,base,x,T,ht,TFcube)
        alpha[t] = 0.5*np.log((1-ht[t,1])/ht[t,1])
        Z[t] = 2*np.sqrt(ht[t,1]*(1-ht[t,1]))
        if t<T-1:
            for i in range(1,m):
                D[t+1,i] = D[t,i]*np.exp(-alpha[t]*h[t,i]*c[i])/Z[t]
    return alpha,Z,D,TFcube,ht,h

def AdaBoost(T,train,test):

    m = len(train.index)
    n = len(train.columns)-1
    h = np.zeros((int(T),m))
    c = np.repeat(0.,m)

    #Define base classifier
    base = []
    for i in range(0,n):
        base.append(np.unique(train.iloc[:,i+1]))
    ht = np.zeros((int(T),4))

    for i in range(0,m):
        c[i] = train.iloc[i,0]
    x = np.matrix(train.iloc[:,1:m])
    ########
    D = np.zeros((T,m))
    D[0,:] = 1/m
    TFcube = base_class(m,n,c,base,x)
    alpha,Z,D,TFcube,ht,h = update(D,m,n,h,c,base,x,T,ht,TFcube)

    ###OutPut Calculation
    m = len(test.index)
    h = np.zeros((T,m))
    for i in range(0,m):
        c[i] = test.iloc[i,0]
    x = np.matrix(test.iloc[:,1:m])
    for t in range(0,T):
        for i in range(0,m):
            #######
            sp = ht[int(t),0]
            ty = ht[int(t),2]
            dim = ht[int(t),3]
            if ty == 0:
                if sp<len(TFcube[int(dim)][0])/2-1:
                    if x[i,int(dim)] <= base[int(dim)][int(sp)]:
                        h[int(t),i] = -1
                    else:
                        h[int(t),i] = 1
                else:
                    h[int(t),i] = -1
            else:
                if sp<len(TFcube[int(dim)][0])/2-1:
                    if x[i,int(dim)] <= base[int(dim)][int(sp)]:
                        h[int(t),i] = 1
                    else:
                        h[int(t),i] = -1
                else:
                    h[int(t),i] = 1

    g = np.matrix(alpha)*np.matrix(h)
    H = g/abs(g)
    currect = 0.
    for i in range(0,m):
        if H[0,i] == c[i]:
            currect+=1
    accuracy = currect/m
    print(accuracy)
    risk = 1-accuracy

    ##Output Table
    t = []
    at = []
    gamma = []
    direc = []
    for i in range(0,T):
        t.append(i+1)
        at.append(ht[i,3])
        if ht[i,0] == 0:
            thres = base[int(at[i])][0]-1
        else:
            thres = base[int(at[i])][0]
        gamma.append(thres)
        if ht[i,2] == 0:
            d = "+1"
        else:
            d = "-1"
        direc.append(d)

    output = pd.DataFrame(np.column_stack((t,at,gamma,direc,alpha)))
    output.columns = ["iteration index","attribute index","threshold","direction","boosting parameter"]

    return risk,accuracy,output,H
#####################

#Define base classifier

###### Main Frame #########
print("Do you want to do n-fold cross validation?(Y/N)")
ans = input().lower()
while ans != "y" and ans != "n":
    print("Y or N only!")
    ans = input().lower()
if ans == "y":
    print("Please Give the Parameter of Cross Validation n:")
    n_cross = input()
    n_cross = int(n_cross)
    Tbest = cross_T(n_cross,trainset)
else:
    print("Please Give the T value:")
    Tbest = int(input())


risk,acc,output,H = AdaBoost(Tbest,trainset,testset)

print("Exit(1) or Show data(2) or Save detail result as csv file(3)?")
answer = int(input())
while answer != 1:
    if answer != 2 and answer !=3:
        print("1 or 2 or 3 only:")
        answer = int(input())
    elif answer ==2:
        print("Tbest = ",Tbest)
        print("Accuracy = ",acc)
        print("Exit(1) or Show data(2) or Save detail result as csv file(3)?")
        answer = int(input())
    elif answer ==3:
        output.to_csv("AdaBoost_hypothesis_header.csv")
        print("Finish saving!")
        print("Exit(1) or Show data(2) or Save detail result as csv file(3)?")
        answer = int(input())
            
                
    
