# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:13:38 2017

@author: xz556
"""
import numpy as np
import time

# parameter for discount facotr "gamma", Upper bound for rc, dc, nc, di, de "rcU,dcU,ncU,diU,deU",
gamma=0.95
rcU=5.0
dcU=5.0
ncU=5.0
diU=5.0
deU=5.0

# Lower and Upper bound for cost "coL,coU", number of points for estimate integration over price "num".
# iteration step size 
coU=7.0
coL=3.0
num=40
iteration=200

# Transition Matrix of lient relationship.
# RelCus_W gives transition matrix when IBM wins the deal
# RelCus_L gives transition matrix when IBM loses the deal
RelCus_W=np.array([[0.2,0.8,0,0,0],[0.1,0.2,0.7,0,0],[0,0.1,0.2,0.7,0],[0,0,0.1,0.2,0.7],[0,0,0,0.3,0.7]])
RelCus_L=np.array([[1,0,0,0,0],[0.5,0.5,0,0,0],[0,0.5,0.5,0,0],[0,0,0.5,0.5,0],[0,0,0,0.5,0.5]])

# pred(...) input deal attributes and price, output probability IBm wins the deal
# dp = 2-p/co which p is the price IBM charges and co is the cost
def pred(rc,dc,nc,di,de,co,dp):
    return (0.001*dc/4+0.03*nc/4+0.04*di/4+0.06*dp+0.1*rc/4+0.2*de/4)/0.431

# costR(...) & costtD(...) input deal attributes and price, output cost function & expectation of cost function
def CostR(rc,dc,nc,di,de,co,dp,w):
    return (1-dp)*co*w
def CostD(rc,dc,nc,di,de,co,dp):
    return CostR(rc,dc,nc,di,de,co,dp,1)*pred(rc,dc,nc,di,de,co,dp)

# Forward(...) output expectation of future cost, E[V(relationship_new)|deal attributes, price]
def Forward(rc,dc,nc,di,de,co,dp,V):
    dist=pred(rc,dc,nc,di,de,co,dp)*RelCus_W[rc]+(1-pred(rc,dc,nc,di,de,co,dp))*RelCus_L[rc]
    return np.dot(V,dist)

# Obj(...) input deal attributes, cost and current value fucntion V, output maximized cost-to-go function over price p
# Obj_D(...) input deal attributes and current value fucntion V, output expectation, over cost, of maximized cost-to-go function
def Obj(rc,dc,nc,di,de,co,V):
    #Obj = CostD(rc,dc,nc,di,de,co,dp)+gamma*Forward(rc,dc,nc,di,de,co,dp,V)
    #Obj is quadratic w.r.t. dp. Obj=a*dp*dp+b*dp+c
    c=CostD(rc,dc,nc,di,de,co,0)+gamma*Forward(rc,dc,nc,di,de,co,0,V)
    a=(CostD(rc,dc,nc,di,de,co,1)+gamma*Forward(rc,dc,nc,di,de,co,1,V)-2*c+CostD(rc,dc,nc,di,de,co,-1)+gamma*Forward(rc,dc,nc,di,de,co,-1,V))/2.0
    b=(CostD(rc,dc,nc,di,de,co,1)+gamma*Forward(rc,dc,nc,di,de,co,1,V)-CostD(rc,dc,nc,di,de,co,-1)-gamma*Forward(rc,dc,nc,di,de,co,-1,V))/2.0
    temp_p=-b/2.0/a
    if temp_p<0:
        opt_p=0
    elif temp_p>1:
        opt_p=1
    else:
        opt_p=temp_p
    return CostD(rc,dc,nc,di,de,co,opt_p)+gamma*Forward(rc,dc,nc,di,de,co,opt_p,V)
def Obj_D(rc,dc,nc,di,de,V):
    step=(coU-coL)/num
    total=0
    for i in range(num):
        total=total+Obj(rc,dc,nc,di,de,10.0**(coL+i*step),V)
    total=total/num
    return total

# T(...) input client relationship rc and current value fucntion V, output updated value function over client relationship
# Bell(...) ---Bellman mapping
def T(rc,V):
    total=0
    for dc in range(int(rcU)):
        for nc in range(int(rcU)):
            for di in range(int(rcU)):
                for de in range(int(rcU)):
                    total=total+Obj_D(rc,dc,nc,di,de,V)
    total=total/(5**4)
    return total
def Bell(V):
    return np.array([T(0,V),T(1,V),T(2,V),T(3,V),T(4,V)])

# Calculate upper(lower) bound of optimal value
start = time.time()
V=np.array([0,0,0,0,0])
W=np.array([2*10**8,2*10**8,2*10**8,2*10**8,2*10**8])
t=list()
tt=list()
t.append(V)
tt.append(W)
temp1=V
temp2=W
for i in range(iteration):
    temp1=Bell(temp1)
    t.append(temp1)
    print(temp1)
    temp2=Bell(temp2)
    tt.append(temp2)
    print(temp2)
end = time.time()
print(end-start)
