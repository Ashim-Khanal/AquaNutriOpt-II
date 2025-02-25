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


# """ Single Objective Time Indexed Model """
# """Create an object of EPA class from the AquaNutriOpt Package"""
# Example = EPA()
# """Generate Optimization inputs directly from WAM outputs.
# See section below for required WAM output files and formats.
# 'P' is the nutrient to be optimized and 3 is the number of
# considered time periods for nutrients loading"""
# Example.WAM_InputGenerator_SO('P', 3)
# """Generate Optimization inputs directly from SWAT outputs.
# See section below for required SWAT output files and formats"""
# Example.SWAT_InputGenerator_SO('N', 3)
# """Read two major input csv files. Names must be Net_Data.csv
# and BMP_Tech.csv. InputGenerator module can create this file.
# Else it can be self created manually without using the InputGenerator
# Module. The format guidelines is provided later in this User Guide"""
# Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 3)
# """Set the available budget for BMPs/TFs for the single objective
# optimization"""
# Example.Set_Cost_Budget(10000)
# """Set the target location where objective nutrient is to be minimized.
# If Input network file has lake as Node 1 then """
# Example.Set_TargetLocation('1')
# """ If we want to bound the other nutrient to a certain value say
# Nitrogen while minimizing Phosphorus is the objective"""
# Example.Set_BoundedMeasures(['N'],[99999])
# """ In order to Set Phosphorus as the Objective"""
# Example.Set_Objective('P')
# """Set the TimeLimit for each optimization problem in seconds.
# In Multiobjective optimization numerous single objective optimization
#  are run"""
# Example.Set_TimeLimit(15)
# """Solve Single Objective Optimization with time periods provided"""
# Example.Solve_SOTI_Det_Model()
#
# Example.Set_Measure_Budget('N',99999)

"""FOR Multi Objective Time Indexed Model (Objectives will be to minimize P, N, Budget)"""

# Example = EPA()
# Example.WAM_InputGenerator_MO(3)
# Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 3)
#
# #default value is set to 10 seconds
# Example.Set_TimeLimit(15) # for solving each MIP optimization problem
#
# # # Setting Budget List (Give Inputs of the budgets you wish to run experiments for)
# Example.Set_Budget_List([0, 100000, 500000, 1000000])
# # # Setting the target node
# Example.Set_TargetLocation('1')
# Example.Solve_MOTI_Det_Model()
# Example.Filter_NonDominatedPoints(0.28)



# """ SWAT Single Objective Time Indexed Model """
# """Create an object of EPA class from the AquaNutriOpt Package"""
# Example = EPA()
# """Generate Optimization inputs directly from WAM outputs.
# See section below for required WAM output files and formats.
# 'P' is the nutrient to be optimized and 3 is the number of
# considered time periods for nutrients loading"""
# Example.SWAT_InputGenerator_SO('P', 22)
# """Generate Optimization inputs directly from SWAT outputs.
# See section below for required SWAT output files and formats"""
# #Example.SWAT_InputGenerator_SO('N', 22)
# """Read two major input csv files. Names must be Net_Data.csv
# and BMP_Tech.csv. InputGenerator module can create this file.
# Else it can be self created manually without using the InputGenerator
# Module. The format guidelines is provided later in this User Guide"""
# Example.Read_Data('Net_Data.csv','BMP_Tech.csv',22)
# """Set the available budget for BMPs/TFs for the single objective
# optimization"""
# Example.Set_Cost_Budget(10000000)
# """Set the target location where objective nutrient is to be minimized.
# If Input network file has lake as Node 1 then """
# Example.Set_TargetLocation('46')
# """ If we want to bound the other nutrient to a certain value say
# Nitrogen while minimizing Phosphorus is the objective"""
# Example.Set_BoundedMeasures(['N'],[999])
# """ In order to Set Phosphorus as the Objective"""
# Example.Set_Objective('P')
# """Set the TimeLimit for each optimization problem in seconds.
# In Multi-objective optimization numerous single objective optimization
#  are run"""
# Example.Set_TimeLimit(15)
# """Solve Single Objective Optimization with time periods provided"""
# Example.Solve_SOTI_Det_Model()
#
# #Example.Set_Measure_Budget('N',99999)


"""FOR SWAT Multi Objective Time Indexed Model (Objectives will be to minimize P, N, Budget)"""

Example = EPA()
#Example.WAM_InputGenerator_MO(22)
Example.SWAT_InputGenerator_MO(22)
Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 22)

#default value is set to 10 seconds
Example.Set_TimeLimit(15) # for solving each MIP optimization problem

# # Setting Budget List (Give Inputs of the budgets you wish to run experiments for)
Example.Set_Budget_List([0, 100000, 500000, 1000000])
# # Setting the target node
Example.Set_TargetLocation('46')
Example.Solve_MOTI_Det_Model()
Example.Filter_NonDominatedPoints(0.28)