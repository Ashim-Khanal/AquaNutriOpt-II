from AquaNutriOpt import *
"""For Multi Objective Time Indexed Model (Objectives will be to minimize P, N, Budget)"""
# An object of class EPA
Example = EPA()
# Read Network, BMP, and timeperiod inputs
Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 5)
# Set the target location for optimization
Example.Set_TargetLocation('1')
# Setting Budget List (Give Inputs of the budgets you wish to run experiments for)
Example.Set_Budget_List([0, 1000000])
# Solving the model
Example.Solve_MOTI_Det_Model()


#Example = EPA()
#Example.Read_Data('Net_Data.csv','BMP_Tech.csv')

#Setting the cost budget
#Example.Set_Cost_Budget(100000)

#Setting the target node


#Example.Set_BoundedMeasures(['N'],[0])
#Example.Set_Objective('P')

#Solving single objective model
#Example.Solve_SO_Det_Model()


""" Single Objective Time Indexed Model """

# Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 5)
#
# Example.Set_TargetLocation('1')
# Example.Set_Objective('N')
# Example.Set_BoundedMeasures(['P'],[9999])
# Example.Solve_SOTI_Det_Model()

#Example.Set_Measure_Budget('N',99999)