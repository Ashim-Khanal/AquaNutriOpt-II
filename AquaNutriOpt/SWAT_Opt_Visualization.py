############
# Creator: 1) Dr. Tarabih, Osama, 2) Dang, Long
# Date: 2025-04-07
# Description: 
# Inputs:
# 1) The Aqua-nutri-opt output file, 
# 2) A pre-defined LOOKUP table containing all possible BMPs and their corresponding LU_Base and LU_Code,
# 3) A LandUse-Subbasin shape file
# Outputs:
# A map of the land use with the optimized BMPs for each subbasin
############

import numpy as np
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from AquaNutriOpt.WAM_Opt_Visualization import visualize_shapefile, visualize_shapefile_to_geojson

def swat_opt_visualization(
    working_path: str,
    aquanutriopt_output_file: str='Res_BMPs_SOTI_100000.txt',
    WAM_lookup_table_file: str = 'LOOKUP_Table.csv',
    SWAT_lookup_table_file: str = 'SWATWAMLUIDs_LOOKUP.csv',
    land_use_subbasin_file: str = 'Land_Subbasin_Intersection.shp',
    watershed_boundary_file: str = 'Watershed.shp',
    out_file_basename: str = 'visualization_output'
    ):
    """
    Visualizes AquaNutriOpt outputs into a map of the land use with optimized BMPs for each subbasin
    
    Args:
        working_path (str): The path to the working directory where the WAM_Opt_Visualization folder will be created.
        aquanutriopt_output_file (str): The name of the AquaNutriOpt output file.
        lookup_table_file (str): The name of the LOOKUP table file that maps BMPs to their corresponding LU_Base and LU_Code.
        land_use_subbasin_file (str): The name of the Land Use Subbasin shapefile.
        watershed_boundary_file (str): The name of the watershed boundary shapefile. Pass the empty string if not available.
        out_file_basename (str): The base name for the output files (image and geojson) (The name of the output file without the extension).
        
    Returns:
        None: The function creates a folder structure and visualizes the data, saving the output as an image and geojson file.
    """
    Wam_path = os.path.join(working_path, 'SWAT_Opt_Visualization')
    Inputs_path = os.path.join(Wam_path, 'Inputs')
    Outputs_path = os.path.join(Wam_path, 'Outputs')

    #create a new folder named 'WAM' under the current working folder if it does not exist
    if not os.path.exists(Wam_path):
        print(f"Create {Wam_path} in the current working directory!")
        os.makedirs(Wam_path)

    #create a new folder named 'Inputs' under the 'WAM' folder if it does not exist
    if not os.path.exists(Inputs_path):
        print(f"Create {Inputs_path} in the current working directory!")
        os.makedirs(Inputs_path)

    #create another new folder named Outputs under the 'WAM' folder if it does not exist
    if not os.path.exists(Outputs_path):
        print(f"Create {Outputs_path} in the current working directory!")
        os.makedirs(Outputs_path)

    # read the watershed shapefile for visualization
    Watershed_path = os.path.join(Inputs_path, 'Watershed')
    Watershed_path = os.path.join(Watershed_path, watershed_boundary_file)
    if watershed_boundary_file.strip() and os.path.exists(Watershed_path):
        Watershed_df = gpd.read_file(Watershed_path)
    else:
        Watershed_df = None

    # Read the Aquanutiopt output file
    # The Aquanutiopt output file contains the optimized BMPs for each subbasin
    # 1,500,000 means a budget constrain and t1 can mean a unit of time.
    df_path = os.path.join(Inputs_path, aquanutriopt_output_file)
    df = pd.read_csv(df_path, 
                    sep=',')

    # Read the master LOOKUP table
    # The LOOKUP table is designed to map BMPs 
    # to their corresponding LU_Base, LU_Code, and BMP_name
    WAM_look_up_tab_path = os.path.join(Inputs_path, WAM_lookup_table_file)
    WAM_LOOKUP_Table = pd.read_csv(WAM_look_up_tab_path)
    WAM_LU_Base = df[' BMPs'].apply(lambda x: WAM_LOOKUP_Table[(WAM_LOOKUP_Table['BMP_Type'] == x)]['LU_Base'].values[0])
    BMP_name = df[' BMPs'].apply(lambda x: WAM_LOOKUP_Table[(WAM_LOOKUP_Table['BMP_Type'] == x)]['BMP_name'].values[0])
    
    
    
    SWAT_look_up_tab_path = os.path.join(Inputs_path, SWAT_lookup_table_file)
    # convert the SWAT_LOOKUP_Table['WAM_LUID'] to integer
    SWAT_LOOKUP_Table = pd.read_csv(SWAT_look_up_tab_path)
    SWAT_LOOKUP_Table['WAM_LUID'] = SWAT_LOOKUP_Table['WAM_LUID'].astype(int)
    SWAT_LOOKUP_Table['landuse_id'] = SWAT_LOOKUP_Table['landuse_id'].astype(int)
    
   

    #for each unique BMPs from LOOKUP_Table,
    # Create a new DataFrame to store the results
    # The DataFrame will include the Subbasin, BMP, LU_Base, and LU_BMP
    Out_df = pd.DataFrame()
    Out_df['Subbasin'] = df['Node']
    Out_df['BMP'] = df[' BMPs']
    Out_df['WAM_LU_Base'] = WAM_LU_Base
    Out_df['BMP_name'] = BMP_name

    # Output for manual check.
    # output_path = os.path.join(Outputs_path, 'test.csv')
    # Out_df.to_csv(output_path)
    # check if the Out_df has the same number of rows as the df
    # print(Out_df.shape[0] == df.shape[0])

    #######################################################################################
    WAM_Inputs_path = os.path.join(Inputs_path, 'Subbasin_Landuse_Intersections')
    vector_data_path = os.path.join(WAM_Inputs_path, land_use_subbasin_file)

    # Read the shapefile
    Existing_LU = gpd.read_file(vector_data_path)

    # print(SWAT_LOOKUP_Table.columns)
    #
    WAM_LUID = np.zeros(len(Existing_LU)) # Initialize to zeros. It means no match.
    # for each gridcode in the Existing_LU DataFrame,
    for i in range(len(Existing_LU)): # nrows = 9
        # iterate through each row in the SWAT_LOOKUP_Table
        for j in range(len(SWAT_LOOKUP_Table)): # nrows = 13
            if Existing_LU['gridcode'].iloc[i] == SWAT_LOOKUP_Table['landuse_id'].iloc[j]:
                WAM_LUID[i] = SWAT_LOOKUP_Table['WAM_LUID'].iloc[j]
                
    
    Existing_LU['WAM_LUID'] = WAM_LUID

    Optimized_LU = Out_df

    

    # print('Stage-1 ', Existing_LU.columns)
    # Existing_LU_path = os.path.join(Outputs_path, 'LU_with_OptimizedBMPs_Stage1.xlsx')
    # Existing_LU.to_excel(Existing_LU_path, index=False)

    # initialize the Opt_BMP_Name nump array with the same length as the Existing_LU DataFrame
    # set all values to "No BMP"
    Opt_BMP_Name = np.full(len(Existing_LU), "No BMP", dtype=object)
    for i in range(len(Existing_LU)):
        for j in range(len(Optimized_LU)):
            if Existing_LU['Subbasin'].iloc[i] == Optimized_LU['Subbasin'].iloc[j]:
                if Existing_LU['WAM_LUID'].iloc[i] == Optimized_LU['WAM_LU_Base'].iloc[j]:
                    Opt_BMP_Name[i] = Optimized_LU['BMP_name'].iloc[j]
    
    Existing_LU['Opt_BMP_Name'] = Opt_BMP_Name
    # Outputs for manual checking

    # print('Stage-2 ',Existing_LU.columns)
    # Existing_LU_path = os.path.join(Outputs_path, 'LU_with_OptimizedBMPs_Stage2.xlsx')
    # Existing_LU.to_excel(Existing_LU_path, index=False)

    # Existing_LU_path = os.path.join(Outputs_path, 'LU_with_OptimizedBMPs.shp')
    # Existing_LU.to_file(Existing_LU_path, 
    #               driver="ESRI Shapefile")

    category_field = "Opt_BMP_Name"
    # category_field =  "Opt_BMP_Na"

    # Define the color mapping for each category
    master_category_color_rule = {
        "drip irrigation": '#FF0000',  # Red
        "FDACS 1": '#00FF00',  # Green
        "FDACS 1 and 2": '#0000FF',  # Blue
        "FDACS 16": '#FFFF00',  # Yellow
        "FDACS 18": '#FF00FF',  # Magenta
        "FDACS 2": '#00FFFF',  # Cyan
        "FDACS 2 and 16": '#FF7F00',  # Orange
        "FDACS 3": '#7F00FF',  # Purple
        "FDACS 32": '#7FFF00',  # Lime
        "FDACS 36": '#FF007F',  # Pink
        "FDACS 4": '#7F7F00',  # Olive
        "FDACS Dairy Sprayfield": '#007F7F',  # Teal
        "FDACS irrigation improvement/fertigation": '#7F007F',  # Maroon
        "FDACS Well or Pipeline": '#00007F',  # Navy
        "fertility and low animal density BMP": '#7F7F7F',  # Gray
        #####################################################
        "fertility and retention BMP": '#FF7FFF',  # Light Pink
        "fertility and storm retention BMP": '#FF7F7F',  # Light Red
        "fertility and waste balance BMP": '#7FFF7F',  # Light Green
        "fertility and water BMP": '#7F7FFF',  # Light Blue
        "fertility BMP": '#FF7F00',  # Light Orange
        "low animal density BMP": '#7F00FF',  # Light Purple
        "non-clear cut BMP": '#FF00FF',  # Light Magenta
        "nutrient balance BMP": '#00FF7F',  # Light Lime
        "Offsite Sewage Deposal": '#7F00FF',  # Light Teal
        "Offsite Sewage Treatment": '#00FFFF',  # Light Cyan
        "Offsite Sewage Treatment and Fertility/storm retention BMPs": '#FF00FF',  # Light Maroon
        "Offsite Tertiary and fertility BMP": '#00FF00',  # Light Navy
        "Offsite Tertiary and storm retention BMP": '#7F7F00',  # Light Olive
        "Offsite Tertiary fertility and storm retention BMP": '#007F7F',  # Light Gray
        "retention and irrigation BMP": '#7F007F',  # Light Pink
        "retention and water BMP": '#00007F',  # Light Red
        ####################Overlapping in color##############################
        "retention BMP": '#00FFFF',  # Light Cyan 
        "Sprayfield": '#FF7FFF',  # Light Pink
        "storm retention BMP": '#7F7F00',  # Light Olive
        "waste balance BMP": '#7F00FF',  # Light Purple
        "water BMP": '#FF00FF',  # Light Magenta
        ###############
        "No BMP": '#FFFFFF',  # White
        
        }
    
    _colors = [
        '#E12300', '#D38306', '#F4ED01', '#649C31', '#038DBB', '#0057D5', '#391B95', '#77239D', '#FE634D',
        '#FFB23C', '#FBF968', '#97D45F', '#00C7FF', '#3689FF', '#5A32EB', '#BC37F2', '#FCB5AF', '#FFD8A8',
        '#FFFCB5', '#CDE9B8', '#90E4FE', '#AAC4FD', '#B28CFB', '#E395FB', '#FE4310', '#FFAA00', '#FDFB44',
        '#75BA43', '#00A2D8', '#0460FF', '#4A24B7', '#9A2ABA', '#B51902', '#AB6702', '#C2BD00', '#4C7C28',
        '#00728B', '#0142A6', '#2C1378', '#63167E', '#FF8C85', '#FFC878', '#FDFA91', '#B0DC87', '#55D6FE',
        '#75A7FE', '#854FFB', '#D456FF', '#FFD8D9', '#FFEBD3', '#FFFCDB', '#E2EBDA', '#CAF2FE', '#D4E4FB',
        '#DACAFB', '#F1C9FE', '#830D00', '#794803', '#888901', '#3B571C', '#004B63', '#013078', '#180B51',
        '#450F5D', '#FF0000', '#FF8000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#8000FF', '#800000',
        '#804000', '#808000', '#008000', '#008080', '#000080', '#400080', '#FF2A00', '#FFAA00', '#AAFF00',
        '#00FF55', '#00AAFF', '#2A00FF', '#8000AA', '#801500', '#805500', '#558000', '#00802A', '#005580',
        '#150080', '#FF5500', '#FFD400', '#55FF00', '#00FFAA', '#0055FF', '#5500FF', '#800055', '#802A00',
        '#806A00', '#2A8000', '#008055', '#002A80', '#2A0080'
    ]

    output_path = os.path.join(Outputs_path, out_file_basename + '.png')
    geojson_output_path = os.path.join(Outputs_path, out_file_basename + '.geojson')

    # Call the function to visualize the Existing_LU shapefile for the category field. 
    # Then, save the figure to the output path.

    # load the shapefile 
    #Existing_LU_path = os.path.join(Outputs_path, 'LU_with_OptimizedBMPs.shp')

    #Existing_LU = gpd.read_file(Existing_LU_path)

    # print(Existing_LU.info())

    visualize_shapefile(Existing_LU, 
                        Watershed_df,
                        category_field, 
                        _colors,  
                        output_path)

    # Call the function to visualize the Existing_LU shapefile for the category field and save it as a GeoJSON file.
    visualize_shapefile_to_geojson(Existing_LU, category_field, _colors, geojson_output_path)


if __name__ == "__main__":
    working_path = os.getcwd()
    swat_opt_visualization(working_path)
