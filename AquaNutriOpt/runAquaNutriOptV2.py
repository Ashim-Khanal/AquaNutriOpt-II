from AquaNutriOpt import *

from AquaNutriOpt.SWAT_Opt_Visualization_new import swat_opt_visualization
from AquaNutriOpt.WAM_Opt_Visualization import wam_opt_visualization
import os 
#
# Example = EPA()
# #
# # """ Single Obj. Model"""
# Example.Read_Data('Net_Data.csv','BMP_Tech.csv')
# #
# # # Setting the target node
# Example.Set_TargetLocation('46')
# #
# Example.Set_BoundedMeasures(['N'], [99999999])
# #
# # # Setting the cost budget
# Example.Set_Cost_Budget(0)
#
# Example.Set_Objective('P')
#
# #Solving single objective model
# Example.Solve_SO_Det_Model()
#
# Example.Get_CorrectedLoad_SO()

# """ Single Objective Time Indexed Model """
def run_wam_single_objective(water_shed_name, model_type, run_visualization):

    # """Create an object of EPA class from the AquaNutriOpt Package"""
    Example = EPA()
    # """Generate Optimization inputs directly from WAM outputs.
    # See section below for required WAM output files and formats.
    # 'P' is the nutrient to be optimized and 3 is the number of
    # considered time periods for nutrients loading"""
    Example.WAM_InputGenerator_SO('P', "1998, 2007, 2017, 2018")
    
    Example.Set_TimeLimit(15)

    Example.Set_TargetLocation('0')

    Example.Set_BoundedMeasures(['N'],[99999])

    Example.Set_Cost_Budget(100000)

    Example.Set_Objective('P')

    Example.Solve_SOTI_Det_Model()
    
    Example.Get_CorrectedLoad_SO()
    Example.Set_Measure_Budget('N',99999)

    if run_visualization:
        wam_opt_visualization(

            working_path=os.getcwd(),                                 # Your working directory, 
            # should have the WAM_Opt_Visualization directory in it

            aquanutriopt_output_file='Res_BMPs_SOTI_100000.txt',    
            # The name of your AquaNutriOpt output file.
            # Will be something like 'Res_BMPs_SOTI_100000.0.txt'. Should be located
            # at <working_path>/WAM_Opt_Visualization/Inputs/<aquanutriopt_output_file>

            lookup_table_file='LOOKUP_Table.csv', # The name of your lookup table file, 
            # located at <working_path>/WAM_Opt_Visualization/Inputs/<lookup_table_file>

            land_use_subbasin_file='Land_Subbasin_Intersection.shp',  
            # The name of your land use 
            # subbasin intersection shapefile, 

                                                                    # located at <working_path>/WAM_Opt_Visualization/Inputs/From_WAM_Model_Generation_Script/<land_use_subbasin_file>

            watershed_boundary_file='Watershed.shp',          
            # The name of your watershed boundary
            # shapefile,

                                                                    # located at <working_path>/WAM_Opt_Visualization/Inputs/Watershed/<watershed_boundary_file>

            out_file_basename='WAM_SO_visualization_output'           # The name of your ouput files 
            #(without the extension).

        )

# """ SWAT Single Objective Time Indexed Model """
# """Create an object of EPA class from the AquaNutriOpt Package"""
def run_swat_single_objective(water_shed_name, model_type, run_visualization):

    if not run_visualization:
        Example = EPA()

        Example.SWAT_InputGenerator_SO('P', '2002, 2003')
        # Example.SWAT_InputGenerator_SO('N', '2002, 2003')

        # """Set the available budget for BMPs/TFs for the single objective
        # optimization"""
        # """Set the target location where objective nutrient is to be minimized.
        # If Input network file has lake as Node 1 then """
        Example.Read_Data('Net_Data.csv','BMP_Tech.csv', 2)
        # """ If we want to bound the other nutrient to a certain value say
        # Nitrogen while minimizing Phosphorus is the objective"""
        Example.Set_TargetLocation('99999')
        
        Example.Set_BoundedMeasures(['N'],[99999])

        Example.Set_Cost_Budget(100000)

        # """ In order to Set Phosphorus as the Objective"""
        Example.Set_Objective('P')
        # """Set the TimeLimit for each optimization problem in seconds.
        # In Multi-objective optimization numerous single objective optimization
        #  are run"""
        Example.Set_TimeLimit(15)
        # # """Solve Single Objective Optimization with time periods provided"""


        Example.Solve_SOTI_Det_Model()

    # Example.Get_CorrectedLoad_SO()
    # Example.Set_Measure_Budget('N',99999)

    # Add this call here, you may have to update parameters as needed
    else:
        swat_opt_visualization(

            working_path=os.getcwd(),                                 # Your working directory, 
            # should have the WAM_Opt_Visualization directory in it

            aquanutriopt_output_file='Res_BMPs_SOTI_100000.txt',    # The name of your AquaNutriOpt output file.
            # aquanutriopt_output_file='BMPs_1500000bound_N_4978.348.txt',
            # Will be something like 'Res_BMPs_SOTI_100000.0.txt'. Should be located
                        # at <working_path>/WAM_Opt_Visualization/Inputs/<aquanutriopt_output_file>

                                                                    # at <working_path>/WAM_Opt_Visualization/Inputs/<aquanutriopt_output_file>
            WAM_lookup_table_file='LOOKUP_Table.csv',
            SWAT_lookup_table_file='SWATWAMLUIDs_LOOKUP.csv',                     # The name of your lookup table file, 
            # located at <working_path>/WAM_Opt_Visualization/Inputs/<lookup_table_file>

            land_use_subbasin_file='Land_Subbasin_Intersection.shp',  # The name of your land use 
            # subbasin intersection shapefile, 
                                                              # located at <working_path>/WAM_Opt_Visualization/Inputs/From_WAM_Model_Generation_Script/<land_use_subbasin_file>
            watershed_boundary_file='Watershed.shp',                  # The name of your watershed boundary
            # shapefile,

                                                                    # located at <working_path>/WAM_Opt_Visualization/Inputs/Watershed/<watershed_boundary_file>
            out_file_basename='SWAT_SO_visualization_output'               # The name of your ouput files 
        #(without the extension).

        )       

        
# create a main function to run the script
if __name__ == "__main__":
    # the function 
    working_path=os.getcwd()
    # Users from the terminal call python runAquaNutriOptV2.py --model SWAT --visualization the type of models and whether they want to run visualization
    water_shed_name = 'Mar Menor Catchment'
    model_type = 'SWAT' # 'EPA_SO', 'EPA_MO',
    run_visualization = True
    if model_type == 'SWAT':
        if run_visualization:
            run_swat_single_objective(water_shed_name, 
                                      model_type, 
                                      run_visualization=True)
        else:
            run_swat_single_objective(water_shed_name,
                                      model_type, 
                                      run_visualization=False)
    
    elif model_type == 'WAM':
        if run_visualization:
            run_wam_single_objective(water_shed_name, 
                                     model_type, 
                                     run_visualization=True)
        else:
            run_wam_single_objective(water_shed_name, 
                                     model_type, 
                                     run_visualization=False)
    elif model_type == 'custom':
        if run_visualization:
            # call your custom model function with visualization
            wam_opt_visualization(

                working_path=os.getcwd(),                                 # Your working directory, 
                # should have the WAM_Opt_Visualization directory in it

                aquanutriopt_output_file='Res_BMPs_SOTI_100000.txt',    
                # The name of your AquaNutriOpt output file.
                # Will be something like 'Res_BMPs_SOTI_100000.0.txt'. Should be located
                # at <working_path>/WAM_Opt_Visualization/Inputs/<aquanutriopt_output_file>

                lookup_table_file='Custom_Model_LOOKUP_Table.csv', # The name of your lookup table file, 
                # located at <working_path>/WAM_Opt_Visualization/Inputs/<lookup_table_file>

                land_use_subbasin_file='Land_Subbasin_Intersection.shp',  
                # The name of your land use 
                # subbasin intersection shapefile, 

                # located at <working_path>/WAM_Opt_Visualization/Inputs/From_WAM_Model_Generation_Script/<land_use_subbasin_file>
                watershed_boundary_file='Watershed.shp',          
                # The name of your watershed boundary
                # shapefile,
                 # located at <working_path>/WAM_Opt_Visualization/Inputs/Watershed/<watershed_boundary_file>
                out_file_basename='Custom_model_SO_visualization_output'           # The name of your ouput files 
            #(without the extension).

        )

    else:
        print("Invalid model type. Please choose from 'WAM', 'SWAT', or 'custom'.")
        exit(1)

