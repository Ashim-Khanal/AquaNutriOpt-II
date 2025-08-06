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

def wam_opt_visualization(
    working_path: str,
    aquanutriopt_output_file: str,
    lookup_table_file: str = 'LOOKUP_Table.csv',
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
    Wam_path = os.path.join(working_path, 'WAM_Opt_Visualization')
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
    # The LOOKUP table is designed to map BMPs to their corresponding LU_Base and LU_Code
    look_up_tab_path = os.path.join(Inputs_path, lookup_table_file)
    LOOKUP_Table = pd.read_csv(look_up_tab_path)
    LU_Base = df[' BMPs'].apply(lambda x: LOOKUP_Table[(LOOKUP_Table['BMP_Type'] == x)]['LU_Base'].values[0])
    LU_BMP = df[' BMPs'].apply(lambda x: LOOKUP_Table[(LOOKUP_Table['BMP_Type'] == x)]['LU_Code'].values[0])
    BMP_name = df[' BMPs'].apply(lambda x: LOOKUP_Table[(LOOKUP_Table['BMP_Type'] == x)]['BMP_name'].values[0])

    #for each unique BMPs from LOOKUP_Table,
    # Create a new DataFrame to store the results
    # The DataFrame will include the Subbasin, BMP, LU_Base, and LU_BMP
    Out_df = pd.DataFrame()
    Out_df['Subbasin'] = df['Node']
    Out_df['BMP'] = df[' BMPs']
    Out_df['LU_Base'] = LU_Base
    Out_df['LU_BMP'] = LU_BMP
    Out_df['BMP_name'] = BMP_name

    # Output for manual check.
    # output_path = os.path.join(Outputs_path, 'BMPs_1500000bound_N_4978.348.csv')
    # Out_df.to_csv(output_path)
    # check if the Out_df has the same number of rows as the df
    # print(Out_df.shape[0] == df.shape[0])

    #######################################################################################
    WAM_Inputs_path = os.path.join(Inputs_path, 'From_WAM_Model_Generation_Script')
    vector_data_path = os.path.join(WAM_Inputs_path, land_use_subbasin_file)

    # Read the shapefile
    Existing_LU = gpd.read_file(vector_data_path)

    Optimized_LU = Out_df

    Opt_LU = np.zeros(len(Existing_LU))
    Opt_BMP = np.zeros(len(Existing_LU))
    #create an empty String array to store the BMP names
    Opt_BMP_Name = np.empty(len(Existing_LU), dtype=object)

    # iterate through each row in the Existing_LU DataFrame, nrows = 6388 
    for i in range (len(Existing_LU)):
        for j in range (len(Optimized_LU)): # nrows = 116
                # iterate through each row in the Optimized_LU DataFrame
                # check if the Subbasin in the Existing_LU DataFrame is equal to the Subbasin in the Optimized_LU DataFrame
                if Existing_LU['B_GRIDCODE'].iloc[i] == Optimized_LU['Subbasin'].iloc[j]:
                    # create a new variable to store the result of checking if the LUID in the Existing_LU DataFrame is equal to the LU_Base in the Optimized_LU DataFrame
                    # cond1 = Existing_LU['LUID'].iloc[i] == Optimized_LU['LU_Base'].iloc[j]
                    # create another variable to store the result of checking if the following command Existing_LU.duplicated(['OBJECTID'], keep=False)[i] equals to True
                    # cond21 = Existing_LU.duplicated(['OBJECTID'], keep=False)[i] == True
                    # create another variable to store the result of the following equal comparison: Existing_LU.loc[Existing_LU['OBJECTID'] == Existing_LU['OBJECTID'].iloc[i]]['LUID'].any() ==  Optimized_LU['LU_Base'].iloc[j]
                    # cond22 = Existing_LU.loc[Existing_LU['OBJECTID'] == Existing_LU['OBJECTID'].iloc[i]]['LUID'].any() ==  Optimized_LU['LU_Base'].iloc[j]
                    # What does it mean if the cond1 is True or the cond21 and cond22 are True

                    # combined = cond1 or (cond21 and cond22)

                    if Existing_LU['LUID'].iloc[i] == Optimized_LU['LU_Base'].iloc[j] \
                    or (Existing_LU.duplicated(['OBJECTID'], keep=False)[i] == True \
                        and Existing_LU.loc[Existing_LU['OBJECTID'] == Existing_LU['OBJECTID'].iloc[i]]['LUID'].any() ==  Optimized_LU['LU_Base'].iloc[j]):
                        
                        Opt_LU[i] = Optimized_LU['LU_BMP'].iloc[j]
                        Opt_BMP[i] = Optimized_LU['BMP'].iloc[j]
                        Opt_BMP_Name[i] = Optimized_LU['BMP_name'].iloc[j]
                        break # exit the inner loop 
                    else:
                        Opt_LU[i] = Existing_LU['LUID'].iloc[i]
                        Opt_BMP[i] = 0
                        Opt_BMP_Name[i] = "No BMP"
                else:
                    Opt_LU[i] = Existing_LU['LUID'].iloc[i]
                    Opt_BMP[i] = 0
                    Opt_BMP_Name[i] = "No BMP"

    # update the Existing_LU DataFrame with the new values of Opt_LU and Opt_BMP
    Existing_LU['Opt_LU'] = Opt_LU
    Existing_LU['Opt_BMP'] = Opt_BMP
    Existing_LU['Opt_BMP_Name'] = Opt_BMP_Name


    # Outputs for manual checking

    # Existing_LU = Existing_LU.drop_duplicates(subset='OBJECTID')
    # Existing_LU_path = os.path.join(Outputs_path, 'LU_with_OptimizedBMPs.xlsx')
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


def visualize_shapefile_to_geojson(gdf, category_field, category_color_rule, output_path=None):
    """
    Visualizes a shapefile with categories mapped to specific colors and saves it as a GeoJSON file.
    
    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the shapefile data.
        category_field (str): The field used for categorization.
        category_color_rule (dict|list): A dictionary mapping category values to colors or a list of colors.
        output_path (str): Path to save the output GeoJSON file. If None, the file is not saved.
        
    Returns:
        None
    """
    # Extract unique categories
    unique_categories = gdf[category_field].unique()

    # Create a color map based on the data
    if isinstance(category_color_rule, list):
        color_map = {}
        for i, category in enumerate(unique_categories):
            color_map[category] = category_color_rule[i % len(category_color_rule)]
        color_map["No BMP"] = '#FFFFFF'  # White for "No BMP"
    else:
        color_map = {category: category_color_rule.get(category, "white") for category in unique_categories}
    
    # Add a 'category' field to the GeoDataFrame
    gdf['category'] = gdf[category_field]

    # Add a 'color' field to the GeoDataFrame based on the category_color_rule
    gdf['color'] = gdf[category_field].map(color_map).fillna('#FFFFFF')
    
    gdf = gdf[gdf['category'] != 'No BMP']
    gdf = gdf[~gdf['category'].isna()]
    
    # Save the GeoDataFrame as a GeoJSON file
    gdf.to_crs('EPSG:4326').to_file(output_path, driver='GeoJSON')

def visualize_shapefile(gdf, 
                        watershed_df,
                        category_field, 
                        category_color_rule, 
                        output_path=None):
    """
    Visualizes a shapefile with categories mapped to specific colors.

    Parameters:
        gdf (GeoPandas Dataframe): The GeoDataFrame containing the shapefile data.
        watershed_df (GeoPandas Dataframe): The GeoDataFrame containing the watershed boundary data. Pass None if not available.
        category_field (str): The field used for categorization such as 'Opt_BMP_Name'.
        category_color_rule (dict|list): A dictionary mapping category values to colors or a list of colors.
        output_path (str): Path to save the output plot. If None, the plot is not saved.

    Returns:
        None
    """
    # Extract unique categories of the specified field
    unique_categories = gdf[category_field].unique()
    # print("Unique categories:", unique_categories)

    # Create a color map based on the data
    if isinstance(category_color_rule, list):
        color_map = {}
        for i, category in enumerate(unique_categories):
            color_map[category] = category_color_rule[i % len(category_color_rule)]
        color_map["No BMP"] = '#FFFFFF'  # White for "No BMP"
    else:
        color_map = {category: category_color_rule.get(category, "white") for category in unique_categories}
    # print("Color map:", color_map)
    # use color and pattern to distinguish "retention BMP"


    # Plot the shapefile using gdf["geometry"]
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the watershed GeoDataFrame
    if watershed_df is not None:
        watershed_df.plot(ax=ax, 
                        color='lightblue', 
                        edgecolor='black', 
                        alpha=0.5
                        )

    gdf.plot(column=category_field, 
             ax=ax, 
             color=gdf[category_field].map(color_map), 
            #  edgecolor='black',
                # linewidth=0.2,
             legend=True)
    
    # add legend into the plot to explain its colors.
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=category,
                            markerfacecolor=color, markersize=10) for category, color in color_map.items() if category != "No BMP"]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title=category_field)
    # Set title and labels
    ##############
    ax.set_title('Land Use Categories')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Save the figure if output_path is provided
    if output_path:
        # Save the figure as a PNG file
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save the legend as a separate image
        legend_path = output_path.replace('.png', '_legend.png')
        save_legend_as_image(color_map, legend_path)

    # Show the plot
    # plt.show()

    # Close the figure
    plt.close(fig)

def save_legend_as_image(color_map, output_path):
    """
    Saves a legend as a standalone image.

    Args:
        color_map (dict): A dictionary mapping category labels to colors.
        output_path (str): Path to save the legend image.

    Returns:
        None
    """
    # Filter out invalid categories and create handles and labels together
    filtered_items = [(category, color) for category, color in color_map.items() if category and str(category).strip() and category != "No BMP"]
    
    # Create handles and labels from the filtered items
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=10) for category, color in filtered_items]
    labels = [category for category, color in filtered_items]

    # Create a new figure for the legend
    fig_legend = plt.figure(figsize=(3, 0.10 * len(filtered_items)))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')  # Turn off the axes

    # Add the legend to the figure
    legend = ax_legend.legend(handles, labels, loc='center', frameon=False)

    # Save the legend as an image
    fig_legend.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Legend saved to {output_path}")

    # Close the figure
    plt.close(fig_legend)


if __name__ == "__main__":
    working_path = os.getcwd()
    wam_opt_visualization(working_path)
