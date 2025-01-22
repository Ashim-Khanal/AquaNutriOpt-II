from AquaNutriOpt import *

#Example= EPA()

""" Single Obj. Model"""
#Example.Read_Data('Net_Data.csv','BMP_Tech.csv')

#Setting the cost budget
#Example.Set_Cost_Budget(100000)

#Setting the target node
#Example.Set_TargetLocation('1')

#Example.Set_BoundedMeasures(['N'], [0])
#Example.Set_Objective('P')

#Solving single objective model
#Example.Solve_SO_Det_Model()


""" Single Objective Time Indexed Model """

#Example = EPA()

#Example.WAM_InputGenerator_SO('P', 3)

#Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 3)

#Example.Set_Cost_Budget(10000)

#Example.Set_TargetLocation('1')

#Example.Set_BoundedMeasures(['N'],[99999])

#Example.Set_Objective('P')

#Example.Set_TimeLimit(15)

#Example.Solve_SOTI_Det_Model()

#Example.Set_Measure_Budget('N',99999)

"""FOR Multi Objective Time Indexed Model (Objectives will be to minimize P, N, Budget)"""

Example = EPA()
Example.WAM_InputGenerator_MO(3)
Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 3)

#default value is set to 10 seconds
Example.Set_TimeLimit(15) # for solving each MIP optimization problem

# # Setting Budget List (Give Inputs of the budgets you wish to run experiments for)
Example.Set_Budget_List([0, 100000, 500000, 1000000])
# # Setting the target node
Example.Set_TargetLocation('1')
Example.Solve_MOTI_Det_Model()
Example.Filter_NonDominatedPoints(0.28)