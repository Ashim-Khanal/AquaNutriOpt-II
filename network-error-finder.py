import numpy as np  
import cplex   
import pulp
import time 

#%% Reading the Inputs
NetFile = open('Net_Data.csv')
NetFile.readline()
NN = []
NNs = []
NNp_i = {}
NNn_i = {}
Pim = {}
BETAij = {}
TTi = {} # Set of all Technologies that can be implemented in loaction i
while True:
    l = NetFile.readline()
    if l=='':
        break
    l = l.split(',')    
    NN.append(l[0])
    if l[1] != '':   
        NNn_i[l[0]] = l[1].split(' ')
    else:
        NNn_i[l[0]] = []
    if l[2] != '':
        NNp_i[l[0]] = l[2].split(' ')
    else:
        NNp_i[l[0]] = [] 
        
    if len(NNp_i[l[0]]) >1:
        NNs.append(l[0])
        temp = l[3].split(' ')
        assert(len(temp) == len(NNp_i[l[0]]) )
        for j in range(len(temp)):
            BETAij[(l[0], NNp_i[l[0]][j])] = float(temp[j])
            
    Pim[(l[0], 'P')] = float(l[4])
    Pim[(l[0], 'N')] = float(l[5])  
    
    
    if l[6] != '\n': 
        l[6] = l[6].strip(' \n')
        l[6] = l[6].strip('\n')
        TTi[l[0]] = l[6].split(' ')
    else:
        TTi[l[0]] = []

    
NetFile.close()
MM = ['N','P'] # Set of all measures

L = '1'
ZZ = ['P'] # Set of Objectives
ZZp = ['N'] # Set of bounded measures


#%% Parameters
TBFile = open('BMP_Tech.csv')
TBFile.readline()
ALPHAtm = {};
ALPHA_HATtm = {};

Cit = {};
# while True:
#     l = TBFile.readline()
#     if l=='':
#         break
#     temp = {}
#     l = l.split(',')
#     # effectiveness
#     for i in range(1,len(l)):
#         if i == CostInd:
#             for j in NN:
#                 if l[0] in TTi[j]:
#                     Cit[(j,l[0])] = float(l[i])
#         else:
#             ind = Header[i].find('_')
#             temp[(Header[i][0:ind],Header[i][ind+1:])] = float(l[i])/100
#     for m in MM:
#         ALPHAtm[(l[0],m)] = (temp[(m,'UB')] + temp[(m,'LB')])/2
#         ALPHA_HATtm[(l[0],m)] = temp[(m,'UB')] - ALPHAtm[(l[0],m)]
                 




Um = {};
Um['N'] = 999
C = 300000;
BigM = 9999

#%% Network Check
for i in NN:
    for j in NNp_i[i]:
        if not (j in NN):
            print("{} in Outgoing of {} is not defined.".format(j,i))
            
            
for i in NN:
    for j in NNn_i[i]:
        if not (j in NN):
            print("{} in Ingoing of {} is not defined.".format(j,i)) 
            
for i in NN:
    for j in NNp_i[i]:
        if j in NN and not(i in NNn_i[j]):
            print(j," is in ", i, "'s Outgoing, but the other way is not correct.")

for i in NN:
    for j in NNn_i[i]:
        if j in NN and not(i in NNp_i[j]):
            print(j," is in ", i, "'s Ingoing, but the other way is not correct.")

LakeCounter = 0           
for i in NN:
    if NNp_i[i] == []:
        LakeCounter += 1
        
if LakeCounter >= 2:
    print("There are more than one node with null outgoing node. It has to be only one which is the lake")
 
for i in NN:
    if len(NNp_i[i])>1:
        for j in range(len(NNp_i[i])):
            if BETAij[(i, NNp_i[i][j])] < 0:
                print('The ratio for {} in spliting point of node {} is negative'.format(NNp_i[i][j], i))