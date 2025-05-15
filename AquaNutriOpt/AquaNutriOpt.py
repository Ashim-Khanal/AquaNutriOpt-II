import numpy as np
import pulp
import os
import zipfile as zp
import sys
import time
import csv
#import math
#from utils import *
import pandas as pd
from AquaNutriOpt.GenInputMO import gen_input_mo
from AquaNutriOpt.GenInputSO import gen_input_so
from AquaNutriOpt.SWAT_Network_Automation_MO import swat_network_automation_mo
from AquaNutriOpt.SWAT_Network_Automation_TN import swat_network_automation_tn
from AquaNutriOpt.SWAT_Network_Automation_TP import swat_network_automation_tp
from AquaNutriOpt.WAM_Network_Automation_MO import wam_network_automation_mo
from AquaNutriOpt.WAM_Network_Automation_TN import WAM_Network_Automation_TN
from AquaNutriOpt.WAM_Network_Automation_TP import WAM_Network_Automation_TP


BigM = 9999
class EPA:
    def __init__(self):
        print('**************************************************************')
        print('**************************************************************')
        print('**** EPA Package -- Version 02.0 *****************************')
        print('**************************************************************')
        print('**************************************************************')
        self.Um = {}
        self.C = np.Inf
        self.ZZ = ['P'] # Default objective is set to minimize the phosphorus
        self.budgetList = []
        self.Writefiles = True
        self.Correction_Factor_P = 0.55  # Average correction factor value for Phosphorus
        self.Correction_Factor_N = 1.05  # Average correction factor value for Nitrogen
        self.timeLimit = 15
        self.softwareCode = 0  # WAM or SWAT selection
        self.CorrectionCode = 1 #
        self.BigBound = 99999999999
        self.setting = 'Filtered'
        self.networkAutomationTimePeriods = None  # Time periods for network automation scripts, None or string such as '2018, 2020'
    # %%
    def Set_TimeLimit(self, timeLimit):   #set timelimit in seconds
        self.timeLimit = timeLimit

    def Solve_SO_Det_Model(self, Budget=np.Inf):

        if Budget == np.Inf:
            pass
        else:
            self.C = Budget
        #self.Solver = Solver
        print('**************************************************************')
        print('**************************************************************')
        print('Building the model ... ***************************************')
        print('**************************************************************')
        print('**************************************************************')
        # Modeling
        MODEL = pulp.LpProblem("Deterministic Model", pulp.LpMinimize)
        # Variables

        Fijm = pulp.LpVariable.dicts('F', [(i, j, m) for m in self.MM for i in self.NN for j in self.NNp_i[i]],
                                     lowBound=0)
        for m in self.MM:
            for i in self.NN:
                for j in self.NNp_i[i]:
                    if i[-1] == 'a':
                        Fijm[(i, j, m)].lowBound = None

        Xit = pulp.LpVariable.dicts('X', {(i, t) for i in self.NN for t in self.TTi[i]}, lowBound=0, upBound=1,
                                    cat=pulp.LpInteger)
        Yijmt = pulp.LpVariable.dicts('Y',
                                      {(i, j, m, t) for j in self.NN for i in self.NNn_i[j] for m in self.MM for t in
                                       self.TTi[j]}, lowBound=0)

        # Objective
        MODEL += pulp.lpSum([Fijm[(i, self.L, self.ZZ[0])] for i in self.NNn_i[self.L]]), 'Obj'

        # Constraints
        ## Cons. 1
        for j in self.NN:
            if len(self.NNp_i[j]) > 1:
                continue
            for m in self.MM:
                if j != self.L:
                    for Time in range(self.TimePeriod):
                        MODEL += (pulp.lpSum([Fijm[(i, j, m)] for i in self.NNn_i[j]])
                                  - pulp.lpSum([Fijm[(j, i, m)] for i in self.NNp_i[j]])
                                  - pulp.lpSum([self.ALPHAtm[(t, m)] * Yijmt[(i, j, m, t)] for t in self.TTi[j] for i in
                                                self.NNn_i[j]])
                                  - pulp.lpSum(
                                    [self.PimTime[(j, m,Time)] * self.ALPHAtm[(t, m)] * Xit[(j, t)] for t in self.TTi[j]]) <= -
                                  self.PimTime[(j, m, Time)]), 'C1_{}_{}'.format(j, m)
        for j in self.NN:
            if len(self.NNp_i[j]) > 1:
                continue
            for i in self.NNn_i[j]:
                for m in self.MM:
                    for t in self.TTi[j]:
                        MODEL += Yijmt[(i, j, m, t)] <= Fijm[(i, j, m)], 'LC1_{}_{}_{}_{}'.format(i, j, m, t)
                        MODEL += Yijmt[(i, j, m, t)] <= BigM * Xit[(j, t)], 'LC2_{}_{}_{}_{}'.format(i, j, m, t)
                        MODEL += Yijmt[(i, j, m, t)] >= Fijm[(i, j, m)] - BigM * (
                                    1 - Xit[(j, t)]), 'LC3_{}_{}_{}_{}'.format(i, j, m, t)

        # Cons. 2
        for i in self.NN:
            MODEL += pulp.lpSum([Xit[(i, t)] for t in self.TTi[i]]) <= 1, 'C2_{}'.format(i)

        # Cons. 3
        for m in self.ZZp:
            MODEL += pulp.lpSum([Fijm[(i, self.L, m)] for i in self.NNn_i[self.L]]) <= self.Um[m], 'C3_{}'.format(m)

        # Cons. 4
        MODEL += pulp.lpSum([self.Cit[(i, t)] * Xit[(i, t)] for i in self.NN for t in self.TTi[i]]) <= self.C, 'C4'

        # Cons. 5
        for j in self.NNs:
            for k in self.NNp_i[j]:
                for m in self.MM:
                    MODEL += Fijm[(j, k, m)] == self.BETAij[(j, k)] * pulp.lpSum(
                        Fijm[(i, j, m)] for i in self.NNn_i[j]), 'C5_{}_{}_{}'.format(j, k, m)

        if self.C == np.Inf:
            print('**WARNING**: No budget is set for cost!!')

        print('**************************************************************')
        print('**************************************************************')
        print('Solving the model ... ****************************************')
        print('**************************************************************')
        print('**************************************************************')


        self.solver = pulp.PULP_CBC_CMD(timeLimit=self.timeLimit, msg=False)
        #solver = pulp.get_solver(self.Solver)
        Sol = MODEL.solve(self.solver)

        print('**************************************************************')
        print('**************************************************************')
        print('Generating the results ... ***********************************')
        print('**************************************************************')
        print('**************************************************************')

        if self.C == 0:
            self.baseLoad = MODEL.objective.value()

        elif self.C != 0:
            self.TargetLoad = MODEL.objective.value()

        file = open('Res_BMPs_SO_{}.txt'.format(self.C), 'w+')
        file.write('Node, BMPs\n')
        Counter = 0
        for i in self.NN:
            for t in self.TTi[i]:
                try:
                    if Xit[(i, t)].value() > .5:
                        ind = i.find('_')
                        if ind > 0:
                            file.write(i[0:ind])
                        else:
                            file.write(i)
                        file.write(',')
                        ind = t.find('_')
                        file.write(t[3:ind])
                        file.write('\n')
                except:
                    Counter = Counter + 1
        file.close()

        file = open('Res_Flow_SO_{}.txt'.format(self.C), 'w+')
        for j in self.NN:
            for i in self.NNn_i[j]:
                try:
                    # print('{} ==> {} : {}'.format(i, j, Fijm[(i,j,'P')].value() ) )

                    file.write('{}_{} {}\n'.format(i, j, str(Fijm[(i, j, 'P')].value())))
                except:
                    pass
        file.close()

        print('******************************************************************************')
        print('******************************************************************************')
        print('Optimization solutions are ready. Find the results in the directory.**********')
        print('******************************************************************************')
        print('******************************************************************************')

    # %%
    def Solve_SOTI_Det_Model(self, Budget=np.Inf):

        if isinstance(self.TimePeriod, int) != True:
            print("Please Enter integer values")
            return "Null"
        elif self.TimePeriod < 1:
            print("Please Enter the value of Time greater or equal to 1")
            return "Null"
        print("time period =", self.TimePeriod)
        if Budget == np.Inf:
            pass
        else:
            self.C = Budget

        #self.Solver = Solver
        print('**************************************************************')
        print('**************************************************************')
        print('Building the model ... ***************************************')
        print('**************************************************************')
        print('**************************************************************')
        # %% Modeling
        MODEL = pulp.LpProblem("Deterministic Time Indexed Model", pulp.LpMinimize)
        # Variables

        FijmTime = pulp.LpVariable.dicts('F', [(i, j, m, Time) for m in self.MM for i in self.NN for j in self.NNp_i[i] for Time in
                                               range(self.TimePeriod)], lowBound=0)
        for m in self.MM:
            for i in self.NN:
                for j in self.NNp_i[i]:
                    for Time in range(self.TimePeriod):
                        if i[-1] == 'a':
                            FijmTime[(i, j, m, Time)].lowBound = None

        Xit = pulp.LpVariable.dicts('X', {(i, t) for i in self.NN for t in self.TTi[i]}, lowBound=0, upBound=1, cat=pulp.LpInteger)
        YijmtTime = pulp.LpVariable.dicts('Y', {(i, j, m, t, Time) for j in self.NN for i in self.NNn_i[j] for m in self.MM for t in self.TTi[j] for Time in range(self.TimePeriod)}, lowBound=0)

        # Objective
        MODEL += pulp.lpSum([FijmTime[(i, self.L, self.ZZ[0], Time)] for i in self.NNn_i[self.L] for Time in range(self.TimePeriod)]), 'Obj'

        # Constraints
        ## Cons. 1
        for j in self.NN:
            if len(self.NNp_i[j]) > 1:
                continue
            for m in self.MM:
                if j != self.L:
                    for Time in range(self.TimePeriod):
                        MODEL += (pulp.lpSum([FijmTime[(i, j, m, Time)] for i in self.NNn_i[j]])
                                  - pulp.lpSum([FijmTime[(j, i, m, Time)] for i in self.NNp_i[j]])
                                  - pulp.lpSum([self.ALPHAtm[(t, m)] * YijmtTime[(i, j, m, t, Time)] for t in self.TTi[j] for i in self.NNn_i[j]])
                                  - pulp.lpSum([self.PimTime[(j, m, Time)] * self.ALPHAtm[(t, m)] * Xit[(j, t)] for t in self.TTi[j]]) <= -self.PimTime[(j, m, Time)]), 'C1_{}_{}_{}'.format(j, m, Time)
        for j in self.NN:
            if len(self.NNp_i[j]) > 1:
                continue
            for i in self.NNn_i[j]:
                for m in self.MM:
                    for t in self.TTi[j]:
                        for Time in range(self.TimePeriod):
                            MODEL += YijmtTime[(i, j, m, t, Time)] <= FijmTime[
                                (i, j, m, Time)], 'LC1_{}_{}_{}_{}_{}'.format(i, j, m, t, Time)
                            MODEL += YijmtTime[(i, j, m, t, Time)] <= BigM * Xit[(j, t)], 'LC2_{}_{}_{}_{}_{}'.format(i,
                                                                                                                      j,
                                                                                                                      m,
                                                                                                                      t,
                                                                                                                      Time)
                            MODEL += YijmtTime[(i, j, m, t, Time)] >= FijmTime[(i, j, m, Time)] - BigM * (
                                        1 - Xit[(j, t)]), 'LC3_{}_{}_{}_{}_{}'.format(i, j, m, t, Time)

        # Cons. 2
        for i in self.NN:
            MODEL += pulp.lpSum([Xit[(i, t)] for t in self.TTi[i]]) <= 1, 'C2_{}'.format(i)

        # Cons. 3
        for m in self.ZZp:
            for Time in range(self.TimePeriod):
                MODEL += pulp.lpSum([FijmTime[(i, self.L, m, Time)] for i in self.NNn_i[self.L]]) <= self.Um[m], 'C3_{}_{}'.format(Time, m)

        # Cons. 4
        MODEL += pulp.lpSum([self.Cit[(i, t)] * Xit[(i, t)] for i in self.NN for t in self.TTi[i]]) <= self.C, 'C4'

        # Cons. 5
        for j in self.NNs:
            for k in self.NNp_i[j]:
                for m in self.MM:
                    for Time in range(self.TimePeriod):
                        MODEL += FijmTime[(j, k, m, Time)] == self.BETAij[(j, k)] * pulp.lpSum(
                            FijmTime[(i, j, m, Time)] for i in self.NNn_i[j]), 'C5_{}_{}_{}_{}'.format(j, k, m, Time)

        if self.C == np.Inf:
            print('**WARNING**: No budget is set !!')

        print('**************************************************************')
        print('**************************************************************')
        print('Solving the model ... ****************************************')
        print('**************************************************************')
        print('**************************************************************')

        self.solver = pulp.PULP_CBC_CMD(timeLimit=self.timeLimit, msg=False)
        #self.solver = pulp.getSolver('PULP_CBC_CMD', timeLimit = self.timeLimit)
        #solver = pulp.get_solver(self.solver)

        Sol = MODEL.solve(self.solver)

        print('**************************************************************')
        print('**************************************************************')
        print('Generating the results ... ***********************************')
        print('**************************************************************')
        print('**************************************************************')

        if Budget != 0:
            self.TargetLoad = MODEL.objective.value()

        if Budget == 0:
            self.baseLoad = MODEL.objective.value()

        if self.Writefiles:
            file = open('Res_BMPs_SOTI_{}.txt'.format(self.C), 'w+')
            file.write('Node, BMPs\n')
            Counter = 0
            for i in self.NN:
                for t in self.TTi[i]:
                    try:
                        if Xit[(i, t)].value() > .5:
                            ind = i.find('_')
                            if ind > 0:
                                file.write(i[0:ind])
                            else:
                                file.write(i)
                            file.write(',')
                            ind = t.find('_')
                            file.write(t[3:ind])
                            file.write('\n')
                    except:
                        Counter = Counter + 1
            file.close()

            file = open('Res_Flow_SOTI_{}.txt'.format(self.C), 'w+')
            for j in self.NN:
                for i in self.NNn_i[j]:
                    try:
                        #print('{} ==> {} : {}'.format(i, j, Fijm[(i,j,'P')].value() ) )
                        file.write('{}_{} {}\n'.format(i,j, str(sum(FijmTime[(i,j,self.ZZ[0],Time)].value() for Time in range(self.TimePeriod)))))
                    except:
                        pass
            file.close()

            print('**************************************************************')
            print('**************************************************************')
            print('Optimization Solution are ready. Find the results in the directory')
            print('**************************************************************')

    def Solve_MOTI_Det_Model(self):

        self.TargetLoad = 0
        self.singleObjResults = []
        sys.setrecursionlimit(3000)
        if self.TimePeriod <= 4:
            BigM = 99999
        else:
            BigM = 999999
        epsilon = 0.000001
        self.Writefiles = False


        # #if self.TimePeriod > 1:
        #     self.MOTI_TL = 10 * math.log(self.TimePeriod)
        #     self.MOTI_loop_TL = 2 * 10 * math.log(self.TimePeriod) * len(self.budgetList)
        # else:
        #     self.MOTI_TL = 10
        #     self.MOTI_loop_TL = 2 * 10 * len(self.budgetList)


        # Setting the target node
        #self.Set_TargetLocation('1')

        current_dir = os.getcwd()
        new_dir = 'MOOResults'
        # Construct the full path to the new folder
        self.MOOPATH = os.path.join(current_dir, new_dir)
        # Create the new folder
        if not os.path.exists(self.MOOPATH):
            os.makedirs(self.MOOPATH)

        file_name = 'singleObjResults.csv'
        file_path = os.path.join(self.MOOPATH, file_name)

        bounddata = open(file_path, 'w+')
        for item in self.MM:
            ZZ = [item]
            self.Set_Objective(item)
            self.Set_BoundedMeasures([itm for itm in self.MM if itm not in ZZ], [999999999 for itm in self.MM if itm not in ZZ ], 'MO')
            for values in self.budgetList:
                self.Solve_SOTI_Det_Model(values)
                if values == 0:
                    self.singleObjResults.append(self.baseLoad)
                else:
                    self.singleObjResults.append(self.TargetLoad)
        self.baseLoad_MO = [self.singleObjResults[0], self.singleObjResults[len(self.budgetList)]]

        bounddata.write('Budget, P, N\n')
        for i in range(len(self.budgetList)):
            bounddata.write(str(self.budgetList[i])+','+str(self.singleObjResults[i])+','+str(self.singleObjResults[i+len(self.budgetList)]))
            if self.budgetList[i] != self.budgetList[-1]:
                bounddata.write('\n')

        self.Writefiles = True
        time.sleep(1)

        """Read the bound data (single objective results) for P and N w.r.t their budgets"""

        bounddata = open(file_path)
        bounddata.readline()  # reading the header (the name of the header is of no concern to the algorithm).
        PhosphorusBound = []
        NitrogenBound = []

        while True:
            bd = bounddata.readline()
            bd = bd.strip('\n')
            if bd == '':
                break
            bd = bd.split(',')
            PhosphorusBound.append((int(bd[0]), float(bd[1])))
            NitrogenBound.append((int(bd[0]), float(bd[2])))

        pulp.PULP_CBC_CMD().msg = 0
        pulp.LpSolverDefault.msg = 0

        allpoints = []
        bounds = []

        print('**************************************************************')
        print('**************************************************************')
        print('Building the model ... ***************************************')
        print('**************************************************************')
        print('**************************************************************')
        # %% Modeling
        MODEL = pulp.LpProblem("Deterministic Time Indexed Model", pulp.LpMinimize)
        # Variables

        FijmTime = pulp.LpVariable.dicts('F', [(i, j, m, Time) for m in self.MM for i in self.NN for j in self.NNp_i[i] for Time in
                                               range(self.TimePeriod)], lowBound=0)
        for m in self.MM:
            for i in self.NN:
                for j in self.NNp_i[i]:
                    for Time in range(self.TimePeriod):
                        if i[-1] == 'a':
                            FijmTime[(i, j, m, Time)].lowBound = None

        Xit = pulp.LpVariable.dicts('X', {(i, t) for i in self.NN for t in self.TTi[i]}, lowBound=0, upBound=1, cat=pulp.LpInteger)
        YijmtTime = pulp.LpVariable.dicts('Y', {(i, j, m, t, Time) for j in self.NN for i in self.NNn_i[j] for m in self.MM for t in self.TTi[j] for Time in range(self.TimePeriod)}, lowBound=0)

        # Constraints
        ## Cons. 1
        for j in self.NN:
            if len(self.NNp_i[j]) > 1:
                continue
            for m in self.MM:
                if j != self.L:
                    for Time in range(self.TimePeriod):
                        MODEL += (pulp.lpSum([FijmTime[(i, j, m, Time)] for i in self.NNn_i[j]])
                                  - pulp.lpSum([FijmTime[(j, i, m, Time)] for i in self.NNp_i[j]])
                                  - pulp.lpSum([self.ALPHAtm[(t, m)] * YijmtTime[(i, j, m, t, Time)] for t in self.TTi[j] for i in self.NNn_i[j]])
                                  - pulp.lpSum([self.PimTime[(j, m, Time)] * self.ALPHAtm[(t, m)] * Xit[(j, t)] for t in self.TTi[j]]) <= -self.PimTime[(j, m, Time)]), 'C1_{}_{}_{}'.format(j, m, Time)
        for j in self.NN:
            if len(self.NNp_i[j]) > 1:
                continue
            for i in self.NNn_i[j]:
                for m in self.MM:
                    for t in self.TTi[j]:
                        for Time in range(self.TimePeriod):
                            MODEL += YijmtTime[(i, j, m, t, Time)] <= FijmTime[
                                (i, j, m, Time)], 'LC1_{}_{}_{}_{}_{}'.format(i, j, m, t, Time)
                            MODEL += YijmtTime[(i, j, m, t, Time)] <= BigM * Xit[(j, t)], 'LC2_{}_{}_{}_{}_{}'.format(i,
                                                                                                                      j,
                                                                                                                      m,
                                                                                                                      t,
                                                                                                                      Time)
                            MODEL += YijmtTime[(i, j, m, t, Time)] >= FijmTime[(i, j, m, Time)] - BigM * (
                                        1 - Xit[(j, t)]), 'LC3_{}_{}_{}_{}_{}'.format(i, j, m, t, Time)

        # Cons. 2
        for i in self.NN:
            MODEL += pulp.lpSum([Xit[(i, t)] for t in self.TTi[i]]) <= 1, 'C2_{}'.format(i)

        # Cons. 5
        for j in self.NNs:
            for k in self.NNp_i[j]:
                for m in self.MM:
                    for Time in range(self.TimePeriod):
                        MODEL += FijmTime[(j, k, m, Time)] == self.BETAij[(j, k)] * pulp.lpSum(
                            FijmTime[(i, j, m, Time)] for i in self.NNn_i[j]), 'C5_{}_{}_{}_{}'.format(j, k, m, Time)

        print('**************************************************************')
        print('**************************************************************')
        print('Solving the model ... ****************************************')
        print('**************************************************************')
        print('**************************************************************')
        self.solver = pulp.PULP_CBC_CMD(timeLimit=self.timeLimit, msg=False)
        #self.solver = pulp.getSolver('PULP_CBC_CMD', timeLimit=self.timeLimit)

        """ Run the multi-objective optimization automation loop """
        file_name = 'allPointsWithBounds.csv'
        file_path = os.path.join(self.MOOPATH, file_name)
        file1 = open(file_path, 'w+')
        for item in self.MM:
            ZZ = [item]  # Set of Objectives
            ZZp = [itm for itm in self.MM if itm not in ZZ]  # Set of bounded measures
            # Objective
            MODEL.setObjective(
                pulp.lpSum([FijmTime[(i, self.L, ZZ[0], Time)] for i in self.NNn_i[self.L] for Time in range(self.TimePeriod)]))

            #start_time = time.time()

            if item == 'P':
                for nb in NitrogenBound:
                    #if time.time() - start_time <= self.MOTI_loop_TL / 2:
                    self.Um[ZZp[0]] = nb[1]
                    budgetLB = nb[0]
                    # Cons. 3
                    for m in ZZp:
                        # MODEL += pulp.lpSum([FijmTime[(i, L, m, Time)] for i in self.NNn_i[L] for Time in range(self.TimePeriod)]) <= (self.Um[m]*(1+epsilon)+epsilon), 'C3'
                        MODEL += pulp.lpSum(
                            [FijmTime[(i, self.L, m, Time)] for i in self.NNn_i[self.L] for Time in range(self.TimePeriod)]) <= self.Um[m]*(1+epsilon), 'C3'
                    file1.write("****************************************************\n")
                    file1.write('Nitrogen is bounded to ' + str(round(self.Um['N']*(1+epsilon),3)) + ' and P is minimized\n')
                    file1.write("****************************************************\n")
                    file1.write('Budget, PhosphporusLoading, NitrogenLoading\n')
                    for budget in NitrogenBound:
                        if budget[0] >= budgetLB:
                            C = budget[0]
                            # Sol = MODEL.solve(pulp.CPLEX_PY())
                            # Cons. 4
                            MODEL += pulp.lpSum([self.Cit[(i, t)] * Xit[(i, t)] for i in self.NN for t in self.TTi[i]]) <= C, 'C4'

                            Sol = MODEL.solve(self.solver)

                            minimalLoadP = MODEL.objective.value()
                            del MODEL.constraints['C4']
                            ## print(MODEL.status)
                            nitrogenValue = sum(
                                FijmTime[(i, self.L, m, Time)].varValue for i in self.NNn_i[self.L] for Time in range(self.TimePeriod))

                            file1.write(str(C) + ',' + str(round(minimalLoadP,3)) + ',' + str(round(nitrogenValue,3)) + '\n')
                            # print('{} in Lake when {} bound is {} for budget {}  = {}'.format('P','N',self.Um['N'], C, MODEL.objective.value()))
                            # print("Nitrogen Value is", nitrogenValue)
                            allpoints.append((C, round(minimalLoadP, 3), round(nitrogenValue, 3)))
                            bounds.append('N_' + str(round(self.Um['N']*(1+epsilon),3)))
                            file_name = 'BMPs_' + str(C) + 'bound_N_' + str(round(self.Um['N']*(1+epsilon),3)) + '.txt'
                            file_path = os.path.join(self.MOOPATH,file_name)
                            BMPfile = open(file_path, 'w+')
                            CounterP = 0
                            BMPfile.write('Node, BMPs\n')
                            for i in self.NN:
                                for t in self.TTi[i]:
                                    try:
                                        if Xit[(i, t)].value() > 0.5:
                                            # print(("{}={}".format(Xit[(i,t)] ,Xit[(i,t)].value())))
                                            ind = i.find('_')
                                            if ind > 0:
                                                BMPfile.write(i[0:ind])
                                            else:
                                                BMPfile.write(i)
                                            BMPfile.write(',')
                                            ind = t.find('_')
                                            BMPfile.write(t[3:ind])
                                            BMPfile.write('\n')
                                            # print('Location i = {}, Technology t = {} => {}'.format(i, t, Xit[(i, t)].value()))
                                    except:
                                        CounterP = CounterP + 1
                                        # print('Location i = {}, Technology t = {} => {}'.format(i, t, 0))
                            BMPfile.close()
                            file_name = 'Flow_P' + str(C) + 'bound_N_' + str(round(self.Um['N']*(1+epsilon),3)) + '.txt'
                            file_path = os.path.join(self.MOOPATH, file_name)
                            Flowfile = open(file_path, 'w+')
                            for j in self.NN:
                                for i in self.NNn_i[j]:
                                    try:
                                        # print('{} ==> {} : {}'.format(i, j, FijmTime[(i,j,'P',Time)].value() ) )
                                        Flowfile.write('{}_{} {}\n'.format(i, j, str(sum(
                                            FijmTime[(i, j, 'P', Time)].value() for Time in range(self.TimePeriod)))))
                                    except:
                                        pass
                                        # print('{} ==> {} : {}'.format(i, j, 0))
                            Flowfile.close()
                    del MODEL.constraints['C3']
                    #else:
                        #print("****Time Limit for Multi-objective optimization loop exceeded****")
                        #break

            elif item == 'N':
                #start_time = time.time()
                for pb in PhosphorusBound:
                    #if time.time() - start_time <= self.MOTI_loop_TL / 2:
                    self.Um[ZZp[0]] = pb[1]
                    budgetLB = pb[0]
                    # Cons. 3
                    for m in ZZp:
                        MODEL += pulp.lpSum(
                            [FijmTime[(i, self.L, m, Time)] for i in self.NNn_i[self.L] for Time in range(self.TimePeriod)]) <= self.Um[m]*(1+epsilon), 'C3'
                    file1.write("******************************************************\n")
                    file1.write('Phosphorous is bounded to ' + str(round(self.Um['P']*(1+epsilon),3)) + ' and N is minimized\n')
                    file1.write("******************************************************\n")
                    file1.write('Budget, PhosphporusLoading, NitrogenLoading\n')
                    for budget in PhosphorusBound:
                        if budget[0] >= budgetLB:
                            C = budget[0]
                            # Sol = MODEL.solve(pulp.CPLEX_PY())
                            # Cons. 4
                            MODEL += pulp.lpSum([self.Cit[(i, t)] * Xit[(i, t)] for i in self.NN for t in self.TTi[i]]) <= C, 'C4'

                            Sol = MODEL.solve(self.solver)

                            minimalLoadN = MODEL.objective.value()
                            del MODEL.constraints['C4']
                            ## print(MODEL.status)
                            phosphorousValue = sum(
                                FijmTime[(i, self.L, m, Time)].varValue for i in self.NNn_i[self.L] for Time in range(self.TimePeriod))
                            file1.write(str(C) + ',' + str(round(phosphorousValue,3)) + ',' + str(round(minimalLoadN,3)) + '\n')
                            allpoints.append((C, round(phosphorousValue, 3), round(minimalLoadN, 3)))
                            bounds.append('P_' + str(round(self.Um['P']*(1+epsilon),3)))
                            # print('{} in Lake when {} bound is {} for budget {}  = {}'.format('N','P',self.Um['P'], C, MODEL.objective.value()))
                            # print("Phosphorous Value is", phosphorousValue)

                            file_name = 'BMPs_' + str(C) + 'bound_P_' + str(round(self.Um['P']*(1+epsilon),3)) + '.txt'
                            file_path = os.path.join(self.MOOPATH, file_name)

                            BMPfile = open(file_path, 'w+')
                            BMPfile.write('Node, BMPs\n')
                            CounterN = 0
                            for i in self.NN:
                                for t in self.TTi[i]:
                                    try:
                                        if Xit[(i, t)].value() > 0.5:
                                            ind = i.find('_')
                                            if ind > 0:
                                                BMPfile.write(i[0:ind])
                                            else:
                                                BMPfile.write(i)
                                            BMPfile.write(',')
                                            ind = t.find('_')
                                            BMPfile.write(t[3:ind])
                                            BMPfile.write('\n')
                                            # print('Location i = {}, Technology t = {} => {}'.format(i, t, Xit[(i, t)].value()))
                                    except:
                                        CounterN = CounterN + 1
                                        # print('Location i = {}, Technology t = {} => {}'.format(i, t, 0))
                            BMPfile.close()

                            file_name = 'Flow_N' + str(C) + 'bound_P_' + str(round(self.Um['P'] * (1 + epsilon), 3)) + '.txt'
                            file_path = os.path.join(self.MOOPATH, file_name)
                            Flowfile = open(file_path, 'w+')
                            for j in self.NN:
                                for i in self.NNn_i[j]:
                                    try:
                                        # print('{} ==> {} : {}'.format(i, j, FijmTime[(i,j,'N',Time)].value() ) )
                                        Flowfile.write('{}_{} {}\n'.format(i, j, str(sum(
                                            FijmTime[(i, j, 'N', Time)].value() for Time in range(self.TimePeriod)))))
                                    except:
                                        pass
                                        # print('{} ==> {} : {}'.format(i, j, 0))
                            Flowfile.close()
                    del MODEL.constraints['C3']
                    #else:
                        #print("****Time Limit for Multi-objective optimization loop exceeded****")
                        #break

        file1.close()

    #def find_non_dominated(points, bounds):
        file_name = 'NonDominatedPoints.csv'
        file_path = os.path.join(self.MOOPATH, file_name)
        file2 = open(file_path, 'w+')
        file2.write('Budget, PhosphorousLoad, NitrogenLoad, FileReference\n')
        non_dominated_points = {}
        # non_dominated = []
        for i in range(len(allpoints)):
            dominated = False
            for j in range(len(allpoints)):
                if i != j and all(allpoints[j][k] <= allpoints[i][k] for k in range(len(allpoints[i])))\
                        and any(allpoints[j][k] < allpoints[i][k] for k in range(len(allpoints[i]))):
                    dominated = True
                    break
            if not dominated:
                non_dominated_points[tuple(allpoints[i])] = bounds[i]
        sorted_dict = dict(sorted(non_dominated_points.items()))
        #sorted_list = sorted(non_dominated_points, key=lambda x: x[0])
        self.non_dominated_points = sorted_dict
        #for i in range(len(non_dominated_points)):
        for point, bound in self.non_dominated_points.items():
            #self.non_dominated_points.append(point)
            file2.write(','.join(map(str, point)) + ',' + 'BMPs_'+str(point[0])+'bound_'+str(bound) + '\n')

        file2.close()
        # return non_dominated


    def Filter_NonDominatedPoints(self, closeness = 0.285):
        file_path = os.path.join(self.MOOPATH, 'NonDominatedPoints.csv')

        # Read the CSV file into a list of rows
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Separate the header and the data
        header = lines[0]  # Save the header
        rows = lines[1:]  # Save the data

        # Sort the rows by the first column (Budget)
        rows.sort(key=lambda x: int(x.split(',')[0]))

        # Overwrite the original file with the sorted data
        with open(file_path, 'w') as file:
            file.write(header)  # Write the header back
            file.writelines(rows)  # Write the sorted rows

        ########################################################################################################
        ################# Filtering CODE, optimal separation found to be around : closeness = 0.285 ############
        data = []
        with open(file_path, 'r') as file:
            # skip the header line
            next(file)

            for line in file:
                # split the line by commas and strip any extra whitespace
                values = line.strip().split(',')
                counter = 0
                for items in values:
                    values[counter] = items.strip()
                    counter += 1
                budget = int(values[0])
                P_load = float(values[1])
                N_load = float(values[2])
                bounds = values[3]

                # Append the row as a tuple to the data list
                data.append((budget, P_load, N_load, bounds))
                #print("*** the requiredData  is ***", data)

        data.sort(key=lambda x: (x[0], x[1]))

        zero_budget_values = [self.singleObjResults[0], self.singleObjResults[len(self.budgetList)]]

        # Convert the sorted data into a dictionary

        result_dict = {
            'Budget': [row[0] for row in data],
            'PhosphorousLoad': [row[1] for row in data],
            'NitrogenLoad': [row[2] for row in data],
            'Bounds': [row[3] for row in data]
        }

        unique_budgets = set(result_dict['Budget'])

        percent_reduction = []
        for val in data:
            percent_reduction.append(((zero_budget_values[0] - val[1]) * 100 / zero_budget_values[0],
                                      (zero_budget_values[1] - val[2]) * 100 / zero_budget_values[1]))
        #print("percent reduction is: ", percent_reduction)

        # print(data[0])
        Budgets = (list(set(result_dict['Budget'])))
        Budgets.sort()
        # print(Budgets)

        filtered_nondominated = []
        temp_index = 0
        counter = 0
        filtered_nondominated.append(data[counter])


        for i in range(len(percent_reduction)):
            try:
                # if skip == True:
                # skip = False
                # continue
                if Budgets[counter] != data[i + 1][0]:
                    #print("Change in Budget")
                    counter += 1
                    filtered_nondominated.append(data[i + 1])
                    temp_index += 1
                    continue

                else:
                    #print("itr {} , list is {}, index {}".format(i, filtered_nondominated, temp_index))

                    index = data.index(filtered_nondominated[temp_index])

                    # if (abs(percent_reduction[index][0]-percent_reduction[i+1][0]) <= closeness) and (abs(percent_reduction[index][1]-percent_reduction[i+1][1]) <= closeness):
                    # del filtered_nondominated[temp_index]

                    # if temp_index == 0:
                    # filtered_nondominated.append(data[i + 2])
                    # skip = True

                    # if temp_index > 0:
                    # temp_index -= 1

                    # print("a false non dominated point in optimization removed")

                    if abs(percent_reduction[index][1] - percent_reduction[i + 1][1]) <= closeness:
                        pass

                    elif abs(percent_reduction[index][0] - percent_reduction[i + 1][0]) <= closeness:
                        filtered_nondominated.append(data[i + 1])
                        del filtered_nondominated[temp_index]

                    else:
                        filtered_nondominated.append(data[i + 1])
                        temp_index += 1
            except:
                pass
                #print("exception occurred at iteration {}".format(i))

        self.filtered_nondominated = filtered_nondominated
        file_name = 'Filtered_NonDominatedPoints.csv'
        file_path = os.path.join(self.MOOPATH, file_name)
        new_file = open(file_path, 'w+')
        new_file.write('Budget, PhosphorousLoad, NitrogenLoad, FileReference\n')

        for items in filtered_nondominated:
            new_file.write(str(items[0]) + ',' + str(items[1]) + ',' + str(items[2]) + ',' + str(items[3]) + '\n')

        #non_dominated_points = find_non_dominated(allpoints, bounds)
    # %%
    def Get_Corrected_Loads_MO(self, setting = 'Filtered'):  # information of points (P, N) loads will be corrected with the aid of base loading zero budget loads.
        self.setting = setting
        correctedLoad = [0 for i in range(len(self.MM))]
        self.CorrectedPointsMO = []
        if self.setting == 'Normal':
            #for point, bound in non_dominated_points.items():
            for point, bound in self.non_dominated_points.items():
                correctedLoad[0] = self.baseLoad_MO[0] * (1 - self.Correction_Factor_P) + point[1] * self.Correction_Factor_P
                correctedLoad[1] = self.baseLoad_MO[1] * (1 - self.Correction_Factor_N) + point[2] * self.Correction_Factor_N
                self.CorrectedPointsMO.append((point[0], round(correctedLoad[0],3), round(correctedLoad[1],3), 'BMPs_'+str(point[0])+'bound_'+ bound))

            with open("MOOResults/CorrectedNDPs.csv", "w") as file:
                #file.write('Use BoundReference column to determine the corrected points with optimization points from NonDominatedPoints file \n')
                file.write('Budget, P_Corrected, N_Corrected, FileReference\n')
                for items in self.CorrectedPointsMO:
                    file.write(','.join(map(str, items)) + '\n')
                    #file.write(str(items[0]) + ',' + str(items[1]) + ',' + str(items[2]) + ',' + str(items[3]) + '\n')

        elif self.setting == 'Filtered':
            for items in self.filtered_nondominated:
                correctedLoad[0] = self.baseLoad_MO[0] * (1 - self.Correction_Factor_P) + items[1] * self.Correction_Factor_P
                correctedLoad[1] = self.baseLoad_MO[1] * (1 - self.Correction_Factor_N) + items[2] * self.Correction_Factor_N
                self.CorrectedPointsMO.append((items[0], round(correctedLoad[0],3), round(correctedLoad[1],3), items[3]))
            with open("MOOResults/Corrected_FilteredNDPs.csv", "w") as file:
                #file.write('Use BoundReference column to determine the corrected points with postprocessed points from FilteredNonDominatedPoints file \n')
                file.write('Budget, P_Corrected, N_Corrected, FileReference\n')
                for items in self.CorrectedPointsMO:
                    file.write(str(items[0]) + ',' + str(items[1]) + ',' + str(items[2]) + ',' + str(items[3]) + '\n')

    def Read_Data(self, Network, BMP_Tech, TimePeriod=1):
        self.TimePeriod = TimePeriod
        if isinstance(self.TimePeriod, int) != True:

            print("Please Enter integer values")
            return "Null"
        elif self.TimePeriod < 1:
            print("Please Enter the value of Time greater or equal to 1")
            return "Null"

        # Network ---------------------------------------------------------------
        if not os.path.exists(Network):
            print('The network file does not exist. Make sure you have entered the right directory')
            return

        if Network[-3:].lower() != 'csv':
            print('The network file must be a .csv file. Suffix is not csv!!!')
            return

        NetFile = open(Network)

        l = NetFile.readline()
        l = l.split(',')
        self.MM = ['P','N']
        print('The list of imported measures is: ', end='')
        #for i in range(4, len(l) - 1):
            #self.MM.append(l[i])
            #print(l[i], ', ', end='')
        print(self.MM)

        self.NN = []
        self.NNs = []
        self.NNp_i = {}
        self.NNn_i = {}
        self.PimTime = {}
        self.BETAij = {}
        self.TTi = {}  # Set of all Technologies that can be implemented in location i
        while True:
            l = NetFile.readline()
            if l == '':
                break
            l = l.split(',')
            self.NN.append(l[0])
            if l[1] != '':
                self.NNn_i[l[0]] = l[1].split(' ')
            else:
                self.NNn_i[l[0]] = []
            if l[2] != '':
                self.NNp_i[l[0]] = l[2].split(' ')
            else:
                self.NNp_i[l[0]] = []

            if len(self.NNp_i[l[0]]) > 1:
                self.NNs.append(l[0])
                temp = l[3].split(' ')
                assert (len(temp) == len(self.NNp_i[l[0]]))
                for j in range(len(temp)):
                    self.BETAij[(l[0], self.NNp_i[l[0]][j])] = float(temp[j])
            for Time in range(self.TimePeriod):
                self.PimTime[(l[0], 'P', Time)] = float(l[4+Time])
                self.PimTime[(l[0], 'N', Time)] = float(l[4+self.TimePeriod+Time])

            if l[4 + 2 * self.TimePeriod] != '\n':
                l[4 + 2 * self.TimePeriod] = l[4 + 2 * self.TimePeriod].strip(' \n')
                l[4 + 2 * self.TimePeriod] = l[4 + 2 * self.TimePeriod].strip('\n')
                self.TTi[l[0]] = l[4 + 2 * self.TimePeriod].split(' ')
            else:
                self.TTi[l[0]] = []

        NetFile.close()

        # Cost and effectiveness ---------------------------------------------------
        if not os.path.exists(BMP_Tech):
            print('The BMP/Technology information file does not exist. Make sure you have entered the right directory')
            return

        if BMP_Tech[-3:].lower() != 'csv':
            print('The BMP_Tech file must be a .csv file. The suffix is not csv!!!')
            return

        TBFile = open(BMP_Tech)
        l = TBFile.readline()
        l = l.strip('\n')
        Header = l.split(',')
        CostInd = 0
        for i in range(1, len(Header)):
            if Header[i].lower() == 'cost':
                CostInd = i
                break
        if CostInd == 0:
            print("Header of file '{}' has no attribute Cost".format(BMP_Tech))
            print(Header)
            return

        self.ALPHAtm = {}
        self.Cit = {}
        self.ALPHA_HATtm = {}
        while True:
            l = TBFile.readline()
            if l == '':
                break
            temp = {}
            l = l.split(',')
            # effectiveness
            for i in range(1, len(l)):
                if i == CostInd:
                    for j in self.NN:
                        if l[0] in self.TTi[j]:
                            self.Cit[(j, l[0])] = float(l[i])
                else:
                    ind = Header[i].find('_')
                    temp[(Header[i][0:ind], Header[i][ind + 1:])] = float(l[i]) / 100
            for m in self.MM:
                self.ALPHAtm[(l[0], m)] = (temp[(m, 'UB')] + temp[(m, 'LB')]) / 2
                self.ALPHA_HATtm[(l[0], m)] = temp[(m, 'UB')] - self.ALPHAtm[(l[0], m)]

        print('--------------------------------------------------------------')
        print('The data was successfully imported ***************************')
        print('--------------------------------------------------------------')

    # %% Set the budgets
    def Set_Cost_Budget(self, C):
        if C < 0:
            print('WARNING: the budget of the cost is negative.')
        self.C = C
        print('--------------------------------------------------------------')
        print('The cost budget was successfully set to {} ****************'.format(C))
        print('--------------------------------------------------------------')

    def Set_Budget_List(self, budgetList):
        if len(budgetList) < 0:
            print("WARNING: no budgets provided")
        if 0 not in budgetList:
            print("WARNING: CANNOT RUN WITHOUT 0 Budget")
            raise ValueError("Please input 0 as a budget for calculating initial bounds")
        validBudgetList = []
        for value in budgetList:
            if isinstance(value, int) or isinstance(value, float):
                validBudgetList.append(value)
            else:
                raise ValueError("Invalid value type in Budget List: {}".format(value))
        self.budgetList = validBudgetList

    # %% Set the upper limit of measures
    def Set_Measure_Budget(self, Measure, Value):
        if type(Measure) == list:
            if (len(Measure) != len(Value)):
                print("ERROR: The number of entered measures and values does not match!!!")
                return
            else:
                if np.any(np.array(Value) < 0):
                    print("ERROR: Budget values cannot be negative")
                    print(Value)
                    return
                i = 0
                for m in Measure:
                    if not m in self.MM:
                        print(self.MM)
                        return
                    self.Um[m] = Value[i]
                    i += 1
        else:
            if not Measure in self.MM:
                print("ERROR: Measure '{}' is not among the imported measures:".format(Measure))
                print(self.MM)
                return
            elif Value < 0:
                print("ERROR: Budget values cannot be negative")
                return
            else:
                self.Um[Measure] = Value

            # %% Set the target location

    def Set_TargetLocation(self, location='1'):
        if not (location in self.NN):
            if (self.NN == []):
                print("No network has been imported yet. Please read the network data using 'Read_Data'")
                return
            else:
                print(
                    "The entered location '{}' does exit not in the imported network. Make sure you enter a string.".format(
                        location))
                return
        self.L = location

    # %% Set the single Objective
    def Set_Objective(self, Objective_Measure):
        if not (Objective_Measure in self.MM):
            if self.MM == []:
                print(
                    "The list of measure has not been imported yet. Please read the network data using 'Read_Data' first.")
                return
            else:
                print(
                    "The entered measure '{}' does not exist in the list of measures. Make sure you enter a string.".format(
                        Objective_Measure))
                print(self.MM)
                return
        self.ZZ = [Objective_Measure]  # Set of Objectives


    # %% Set the limits of the bounded objectives
    def Set_BoundedMeasures(self, Measures, Bounds, obj = 'SO'):

        if obj == 'SO':
            if not (IsSubset(Measures, self.MM)):
                if self.MM == []:
                    print(
                        "The list of measure has not been imported yet. Please read the network data using 'Read_Data' first.")
                    return
                else:
                    print(
                        "At least one of the entered measures '{}' does not exit in the list of measures. Make sure you enter a string.".format(
                            Measures))
                    print(self.MM)
                    return

            self.ZZp = Measures  # Set of bounded measures
            self.Um = {}
            for i in range(len(self.ZZp)):
                self.Um[self.ZZp[i]] = self.BigBound

            self.Corrected_Bound(Bounds)

        elif obj == 'MO':
            if not (IsSubset(Measures, self.MM)):
                if self.MM == []:
                    print(
                        "The list of measure has not been imported yet. Please read the network data using 'Read_Data' first.")
                    return
                else:
                    print(
                        "At least one of the entered measures '{}' does not exist in the list of measures. Make sure you enter a string.".format(
                            Measures))
                    print(self.MM)
                    return

            self.ZZp = Measures  # Set of bounded measures
            self.Um = {}
            for i in range(len(self.ZZp)):
                self.Um[self.ZZp[i]] = Bounds[i]

        # %% Set the solver
        def Set_Solver(solver):
            if not solver in pulp.listSolvers(onlyAvailable=True):
                print('ERROR: solver {} is not available on your system!'.format(solver))
                return
            self.Solver = solver
            print('Solver is properly set to {}'.format(solver))

    def WAM_InputGenerator_SO(self, Objective, TimePeriod: str = '1995'):
        self.softwareCode = 0
        self.TimePeriod = self.TimePeriodFromString(TimePeriod)
        self.networkAutomationTimePeriods = TimePeriod
        if isinstance(self.TimePeriod, int) != True:
            print("Please Enter integer values")
            return "Null"
        elif self.TimePeriod < 1:
            print("Please Enter the value of TimePeriod starting from equal or greater than 1 in integers")
            return "Null"

        """Set the objective for Input Generator."""
        if Objective not in ['P', 'N']:
            raise ValueError("Invalid objective. Choose 'P' or 'N'")
        self.Objective = Objective

        """Perform actions based on the chosen objective."""
        if self.Objective == 'P':
            self.run_scriptTP()  # Call script or function for objective 1
        elif self.Objective == 'N':
            self.run_scriptTN()  # Call script or function for objective 2
        else:
            raise ValueError("Objective for Input Generator set incorrectly.")

        self.BMP_Selection()

        if self.Objective == 'P':
            """ write code to change the format of network file"""
            if os.path.exists("NetworkInfo.csv"):
                os.remove("NetworkInfo.csv")
            os.rename("WAM/Outputs/WAM_final_output_single_obj_optim_TP.csv", "NetworkInfo.csv")
            #data = pd.read_csv("WAM/Outputs/WAM_final_output_single_obj_optim_TP.csv",dtype=object)
            #output_file = "NetworkInfo.csv"
            #n_rows = len(data)
            #start_index = 4+self.TimePeriod
            #columns_to_add = self.TimePeriod

            #new_columns = {f"N_{i}": 0 for i in range(columns_to_add)}
            #new_data = pd.DataFrame({col: [value] * n_rows for col, value in new_columns.items()})
            # Insert new columns into the DataFrame at the specified position
            #columns = data.columns.tolist()
            # Convert the list to a DataFrame
            #updated_columns = columns[:start_index] + list(new_data.columns) + columns[start_index:]
            #result = pd.concat([data, new_data], axis=1)[updated_columns]
            # Save the updated DataFrame to a new CSV file
            #result.to_csv(output_file, index=False)
            #columns_df = pd.DataFrame([columns])
            #columns_df.to_csv(output_file, index=False)

        elif self.Objective == 'N':
            if os.path.exists("NetworkInfo.csv"):
                os.remove("NetworkInfo.csv")
            os.rename("WAM/Outputs/WAM_final_output_single_obj_optim_TN.csv", "NetworkInfo.csv")
            # """ write code to change the format of network file"""
            # data = pd.read_csv("WAM/Outputs/WAM_final_output_single_obj_optim_TN.csv",dtype=object)
            # output_file = "NetworkInfo.csv"
            #n_rows = len(data)
            #start_index = 4
            #columns_to_add = self.TimePeriod
            #new_columns = {f"P_{i}": 0 for i in range(columns_to_add)}
            #new_data = pd.DataFrame({col: [value] * n_rows for col, value in new_columns.items()})
            # Insert new columns into the DataFrame at the specified position
            # columns = data.columns.tolist()
            #updated_columns = columns[:start_index] + list(new_data.columns) + columns[start_index:]
            #result = pd.concat([data, new_data], axis=1)[updated_columns]
            # Save the updated DataFrame to a new CSV file
            #result.to_csv(output_file, index=False)
            # columns_df = pd.DataFrame([columns])
            # columns_df.to_csv(output_file, index=False)
        else:
            raise ValueError("The objective nutrient for input network generation not set!")

        self.run_GenInputSO()

        print("*****INPUT FILES GENERATED********")

    def WAM_InputGenerator_MO(self, TimePeriod: str = '1995'):
        self.softwareCode = 0
        self.TimePeriod = self.TimePeriodFromString(TimePeriod)
        self.networkAutomationTimePeriods = TimePeriod
        wam_network_automation_mo(os.getcwd(), self.networkAutomationTimePeriods)
        self.BMP_Selection_MO()
        data = "WAM/Outputs/WAM_final_output_multiple_obj_optim.csv"
        output_file = "NetworkInfo.csv"
        # Open the source file in read mode and the destination file in write mode
        with open(data, 'r') as src, open(output_file, 'w') as dest:
            # Read from the source and write to the destination
            for line in src:
                dest.write(line)
        self.run_GenInputMO()

    def SWAT_InputGenerator_MO(self, TimePeriod: str = '1995'):
        self.softwareCode = 1
        self.TimePeriod = self.TimePeriodFromString(TimePeriod)
        self.networkAutomationTimePeriods = TimePeriod
        swat_network_automation_mo(os.getcwd(), self.networkAutomationTimePeriods)
        self.BMP_Selection_MO()
        data = "SWAT/Outputs/SWAT_final_output_multiple_obj_optim.csv"
        output_file = "NetworkInfo.csv"
        # Open the source file in read mode and the destination file in write mode
        with open(data, 'r') as src, open(output_file, 'w') as dest:
            # Read from the source and write to the destination
            for line in src:
                dest.write(line)
        self.run_GenInputMO()

    def SWAT_InputGenerator_SO(self, Objective, TimePeriod: str = '1995'):
        self.softwareCode = 1
        self.TimePeriod = self.TimePeriodFromString(TimePeriod)
        self.networkAutomationTimePeriods = TimePeriod
        if isinstance(self.TimePeriod, int) != True:
            print("Please Enter integer values")
            return "Null"
        elif self.TimePeriod < 1:
            print("Please Enter the value of TimePeriod equal to or greater than 1 in integers")
            return "Null"

        """Set the objective for Input Generator."""
        if Objective not in ['P', 'N']:
            raise ValueError("Invalid objective. Choose either 'P' or 'N'")
        self.Objective = Objective

        """Perform actions based on the chosen objective."""
        if self.Objective == 'P':
            self.run_scriptTP()  # Call script or function for objective 1
        elif self.Objective == 'N':
            self.run_scriptTN()  # Call script or function for objective 2
        else:
            raise ValueError("Objective for Input Generator set incorrectly.")

        self.BMP_Selection()

        if self.Objective == 'P':
            if os.path.exists("NetworkInfo.csv"):
                os.remove("NetworkInfo.csv")
            os.rename("SWAT/Outputs/SWAT_final_output_single_obj_optim_TP.csv", "NetworkInfo.csv")
            """ write code to change the format of network file"""
            # data = pd.read_csv("SWAT/Outputs/SWAT_final_output_single_obj_optim_TP.csv",dtype=object)
            # output_file = "NetworkInfo.csv"
            #n_rows = len(data)
            #start_index = 4 + self.TimePeriod
            #columns_to_add = self.TimePeriod

            #new_columns = {f"N_{i}": 0 for i in range(columns_to_add)}
            #new_data = pd.DataFrame({col: [value] * n_rows for col, value in new_columns.items()})
            # Insert new columns into the DataFrame at the specified position
            # columns = data.columns.tolist()
            #updated_columns = columns[:start_index] + list(new_data.columns) + columns[start_index:]
            #result = pd.concat([data, new_data], axis=1)[updated_columns]
            # Save the updated DataFrame to a new CSV file
            #result.to_csv(output_file, index=False)
            # columns_df = pd.DataFrame([columns])
            # columns_df.to_csv(output_file, index=False)

        elif self.Objective == 'N':
            if os.path.exists("NetworkInfo.csv"):
                os.remove("NetworkInfo.csv")
            os.rename("SWAT/Outputs/SWAT_final_output_single_obj_optim_TN.csv", "NetworkInfo.csv")
            """ write code to change the format of network file"""
            # data = pd.read_csv("SWAT/Outputs/SWAT_final_output_single_obj_optim_TN.csv",dtype=object)
            # output_file = "NetworkInfo.csv"
            #n_rows = len(data)
            #start_index = 4
            #columns_to_add = self.TimePeriod
            #new_columns = {f"P_{i}": 0 for i in range(columns_to_add)}
            #new_data = pd.DataFrame({col: [value] * n_rows for col, value in new_columns.items()})
            # Insert new columns into the DataFrame at the specified position
            # columns = data.columns.tolist()
            #updated_columns = columns[:start_index] + list(new_data.columns) + columns[start_index:]
            #result = pd.concat([data, new_data], axis=1)[updated_columns]
            # Save the updated DataFrame to a new CSV file
            #result.to_csv(output_file, index=False)
            # columns_df = pd.DataFrame([columns])
            # columns_df.to_csv(output_file, index=False)
        else:
            raise ValueError("The objective nutrient for input network generation not set!")

        self.run_GenInputSO()

        print("*****SWAT INPUT FILES GENERATED********")

    def run_GenInputMO(self):
        gen_input_mo(self.TimePeriod)

    #### Osama and Long's Script
    def run_scriptTP(self):
        if self.softwareCode == 0:
            WAM_Network_Automation_TP(os.getcwd(), self.networkAutomationTimePeriods)
        elif self.softwareCode == 1:
            swat_network_automation_tp(os.getcwd(), self.networkAutomationTimePeriods)
            


    def run_scriptTN(self):
        if self.softwareCode == 0:
            WAM_Network_Automation_TN(os.getcwd(), self.networkAutomationTimePeriods)
        elif self.softwareCode == 1:
            swat_network_automation_tn(os.getcwd(), self.networkAutomationTimePeriods)
    ######
    def run_GenInputSO(self):
        gen_input_so(self.TimePeriod)



    def BMP_Selection(self): # jiayi's Code
        # %%
        # Load the data from the uploaded files
        usace_bmp_path = 'BMP_database/USACE_BMP_database.csv'
        usace_bmp_df = pd.read_csv(usace_bmp_path)
        if self.softwareCode == 0:
            if self.Objective == 'P':
                wam_luid_path = 'WAM/Outputs/WAM_unique_LUID_optim_TP.csv'
            elif self.Objective == 'N':
                wam_luid_path = 'WAM/Outputs/WAM_unique_LUID_optim_TN.csv'
            else:
                raise ValueError("File can not be created. Input to Objective for WAM_InputGenerator_SO")
            wam_luid_df = pd.read_csv(wam_luid_path)
            selected_luids = wam_luid_df['LUID']
        elif self.softwareCode == 1:
            if self.Objective == 'P':
                swat_luid_path = 'SWAT/Outputs/SWAT_unique_LUID_optim_TP.csv'     ### SWAT LUIDS are automatically converted to WAM LUIDs using LU names
            elif self.Objective == 'N':
                swat_luid_path = 'SWAT/Outputs/SWAT_unique_LUID_optim_TN.csv'
            else:
                raise ValueError("File can not be created. Input to Objective for SWAT_InputGenerator_SO")
            swat_luid_df = pd.read_csv(swat_luid_path)
            selected_luids = swat_luid_df['LUID']

        # %%
        # Filter BMPs based on the LUIDs provided in the WAM_unique_LUID_optim_TN file
        filtered_bmps = usace_bmp_df[usace_bmp_df['LU_CODE'].isin(selected_luids)]
        # %%
        # Save the filtered BMPs to a new CSV file (optional)
        try:
            filtered_bmps.to_csv('BMPsInfo.csv', index=False)
        except:
            print("BMPsInfo can not be created")
        # Display the filtered BMPs
        # print(filtered_bmps)
        # %%
    def BMP_Selection_MO(self): # jiayi's Code
        # %%
        try:
            usace_bmp_path = 'BMP_database/USACE_BMP_database.csv'
            usace_bmp_df = pd.read_csv(usace_bmp_path)
            if self.softwareCode == 0:
                wam_luid_path = 'WAM/Outputs/WAM_unique_LUID_multiple_obj_optim.csv'
                luid_df = pd.read_csv(wam_luid_path)
            elif self.softwareCode == 1:
                swat_luid_path = 'SWAT/Outputs/SWAT_unique_LUID_multiple_obj_optim.csv'
                luid_df = pd.read_csv(swat_luid_path)
            else:
                print(" We can only work with either Custom Optimization Inputs or WAM and SWAT outputs as the Inputs")
            selected_luids = luid_df['LUID']
        except:
            raise ValueError("File can not be created. Error while creating input to Objective for WAM or SWAT InputGenerator_SO")

        # %%
        # Filter BMPs based on the LUIDs provided in the WAM_unique_LUID_optim_TN file

        filtered_bmps = usace_bmp_df[usace_bmp_df['LU_CODE'].isin(selected_luids)]
        # %%
        # Save the filtered BMPs to a new CSV file (optional)
        try:
            filtered_bmps.to_csv('BMPsInfo.csv', index=False)
        except:
            print("BMPsInfo can not be created")

    def Corrected_Bound(self, Bounds): #User gives the bound corresponding to WAM/SWAT simulations and this functions converts to optimization bounds

        save_dir = "Corrected_Loads_and_Bounds_SO"
        os.makedirs(save_dir, exist_ok=True)

        budget = self.C
        objective = self.ZZ[0]
        #for items in self.ZZp:  # limit people to put only 1 bounded measure
        #self.Corrected_boundLoad = [0 for i in range(len(self.ZZp))]

        self.Writefiles = False     #it does not write the output files for zero budget run
        self.Set_Objective(self.ZZp[0])
        self.Solve_SOTI_Det_Model(Budget=0)
        self.Um = {}
        self.Writefiles = True

        Corrected_boundLoad = Bounds

        for i in range(len(Bounds)):
            if self.ZZp[i] == 'P':
                Corrected_boundLoad[i] = (Bounds[i] + self.baseLoad * (self.Correction_Factor_P - 1)) / self.Correction_Factor_P
            elif self.ZZp[i] == 'N':
                Corrected_boundLoad[i] = (Bounds[i] + self.baseLoad * (self.Correction_Factor_N - 1)) / self.Correction_Factor_N
            else:
                print("***** Please put the correct nutrient *****")
                Corrected_boundLoad[i] = Bounds[i]

        self.C = budget
        self.ZZ[0] = objective

        for i in range(len(Corrected_boundLoad)):
            self.Um[self.ZZp[i]] = Corrected_boundLoad[i]

        #file = open(save_dir + '/' + 'correctedBound.txt', 'w+')

        #for i in range(len(self.ZZp)):
            #file.write('The corrected Bound for the bounded nutrient {} loading at the Target Node was tuned to {}'.format(self.ZZp[i], Corrected_boundLoad))
        #print(" The corrected bound values are {}".format(Corrected_boundLoad))
        #file.close()


    def Get_CorrectedLoad_SO(self): # corrected nutrient load to be optimized at the target

        print("**************************************")
        print("***** Getting Corrected Load *********")
        print("**************************************")

        save_dir = "Corrected_Loads_and_Bounds_SO"
        os.makedirs(save_dir, exist_ok=True)
        filename = 'correctedLoad.txt'
        filepath = os.path.join(save_dir, filename)

        if self.C == 0:
            file = open(filepath, 'w+')
            file.write('The corrected optimized {} nutrient loading at the Target Node is {}'.format(self.ZZ[0], self.baseLoad))
            file.close()

        elif self.C != 0:
            budget = self.C

            self.Writefiles = False

            self.ZZp = [itm for itm in self.MM if itm not in self.ZZ]  # Set of bounded measures
            self.Um = {}
            for i in range(len(self.ZZp)):
                self.Um[self.ZZp[i]] = self.BigBound

            self.Solve_SOTI_Det_Model(Budget=0)

            if self.ZZ[0] == 'P':
                self.correctedLoad = self.baseLoad * (1 - self.Correction_Factor_P) + self.TargetLoad * self.Correction_Factor_P

            elif self.ZZ[0] == 'N':
                self.correctedLoad = self.baseLoad * (1 - self.Correction_Factor_N) + self.TargetLoad * self.Correction_Factor_N


            self.C = budget

            file = open(save_dir + '/' + 'correctedLoad.txt', 'w+')
            file.write('The corrected optimized {} nutrient loading at the Target Node is {}'.format(self.ZZ[0], self.correctedLoad))
            file.close()

    def GetEssential_MOoutputs(self):
        #print("current working dir = ", os.getcwd())
        files_to_zip = []

        if self.setting == 'Filtered':
            csv_path = 'MOOResults/Corrected_FilteredNDPs.csv'
            archiveCode = 0
        elif self.setting == 'Normal':
            csv_path = 'MOOResults/CorrectedNDPs.csv'
            archiveCode = 1
        else:
            print("csv path required for zipping not found")
            return
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                file_name = row[' FileReference'].strip()
                full_path = os.path.join('MOOResults', file_name)
                #if os.path.isfile(full_path):
                    #files_to_zip.append(full_path)
                files_to_zip.append(str(full_path)+'.txt')
                #else:
                    #print(f"** File not found ** : {full_path+'.txt'} - Skipping.")
        #print("files to zip ", files_to_zip)

        with zp.ZipFile('MultiObjResultsEssential.zip','w', compression=zp.ZIP_DEFLATED) as zf:
            try:
                zf.write('MOOResults/allPointsWithBounds.csv')
            except FileNotFoundError:
                print("allPointsWithBounds.csv not found, skipping")

            for file_path in files_to_zip:
                try:
                    zf.write(file_path)
                except FileNotFoundError:
                    print(f"File not found during zipping: {file_path} - skipping")
            try:
                if archiveCode == 0:
                    zf.write('MOOResults/Filtered_NonDominatedPoints.csv')
                    zf.write('MOOResults/Corrected_FilteredNDPs.csv')
                elif archiveCode == 1:
                    zf.write('MOOResults/NonDominatedPoints.csv')
                    zf.write('MOOResults/CorrectedNDPs.csv')
            except:
                print("Non Dominated Points file not found.")
    
    def TimePeriodFromString(self, time_period_string: str) -> int:
        """
        Parses a string of comma-separated years and returns the number of time periods
        
        Args:
            time_periods_string (str): A string containing comma-separated time periods (e.g., "1995,1996,1997").
            
        Returns:
            int: The number of time periods in the string.
        """
        time_period_string = time_period_string.replace(" ", "")
        time_period_string = time_period_string.rstrip(",")
        years = time_period_string.split(',')
        return len(years)


# %%
def IsSubset(X, Y):
    return len(np.setdiff1d(X, Y)) == 0