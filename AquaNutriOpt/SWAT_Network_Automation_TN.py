import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import datetime as dt
import numpy as np
import csv
import sys
from AquaNutriOpt.SWAT_utils import *


def swat_network_automation_tn(working_path: str, time_periods: str):
    """Performs network automation for single-objective optimization on SWAT inputs for minimizing nitrogen.
    
    Args:
        working_path (str): The path to the working directory where WAM inputs and outputs are stored.
        time_periods (str): Time periods to process (e.g., "2018", "2018, 2020").
    """
    ########### Notes ################
    # 1. Need to check LUID column of the final output for any blank value. If yes, need to update the input file. (LUID.xlsx.)
    ##################################


    Swat_path = os.path.join(working_path, 'SWAT')
    Inputs_path = os.path.join(Swat_path, 'Inputs')
    Outputs_path = os.path.join(Swat_path, 'Outputs')


    if not os.path.exists(Swat_path):
        print(f"Create {Swat_path} in the current working directory!")
        os.makedirs(Swat_path)

    #create a new folder named 'Inputs' under the 'WAM' folder if it does not exist
    if not os.path.exists(Inputs_path):
        print(f"Create {Inputs_path} in the current working directory!")
        os.makedirs(Inputs_path)

    # data for computing TP, TN
    #input_TP_filename = 'rch.xlsx'
    input_TN_filename = 'SWAT_rch.xlsx'
    input_reach_in_out_file = 'SWAT_Reach.xlsx'
    input_annual_subbasin_filename = 'SWAT_hru.xlsx' 
    #input_annual_subbasin_filename = 'hru_small_sub01.xlsx'
    input_luid_filename = 'LUIDs.xlsx'


    #subbasin_TP_input_file = os.path.join(Inputs_path, input_TP_filename)
    subbasin_TN_input_file = os.path.join(Inputs_path, input_TN_filename)
    input_luid_file = os.path.join(Inputs_path, input_luid_filename)

    # data for create the reach graph
    data_dir = 'SWAT/Inputs/Reaches'
    if not os.path.exists(os.path.join(working_path, data_dir)):
        print(f"Error: Directory '{data_dir}' does not exist in the current working directory!")
        sys.exit("Exiting program.")
    data_dir = os.path.join(working_path, data_dir)

    # create output folder
    if not os.path.exists(Outputs_path):
        print(f"Create {Outputs_path} in the current working directory!")
        os.makedirs(Outputs_path)

    ##################### 2. Compute the annual flow (not necessary because the data is already available in the SWAT model)

    ################################# 3. Represent a graph of reaches and search the graph for the Ingoing and Outgoing reaches using Reach.xlsx

    input_reach_in_out_file = os.path.join(data_dir, input_reach_in_out_file)
    # print(input_reach_in_out_file)
    my_df = load_data(input_reach_in_out_file)

    from_node_array = my_df['FROM_NODE'].values

    # print(f"Number of vertices {len(from_node_array)}\n")

    # a list of downstream (DS) reachers - define to_node_array storing 'TO_NODE' column
    to_node_array = my_df['TO_NODE'].values
    # print(f"Number of to_nodes {len(to_node_array)}\n")

    # define reach_array storing 'OBJECTID' column
    reach_array = my_df['OBJECTID'].values
    # print(f"Number of reaches {len(reach_array)}\n")

    # Create a set from 'FROM_NODE' and 'TO_NODE' arrays
    nodes_array = np.union1d(from_node_array, to_node_array)

    # print('length ', len(nodes_array), 'first ', nodes_array[0], 'last ', nodes_array[-1], 'data element ',type(nodes_array[0]))

    # ## Create a list of dictionary called 'nodes'
    # Initialize an empty list to store dictionaries
    nodes = []
    # Iterate over the numpy array
    for node in nodes_array:
        # Create a dictionary with 'id' key and corresponding string value
        dict_entry = {'id': str(node)}
        # Append the dictionary to the list
        nodes.append(dict_entry)

    # Print the list of dictionaries
    # print(nodes)

    # ## Create a list of dictionary called 'links'

    # Initialize an empty list to store dictionaries
    links = []

    # Iterate over the numpy array
    for reach, from_node, to_node in zip(reach_array, from_node_array, to_node_array):
        # Create a dictionary with 'id' key and corresponding string value
        dict_entry = {"source": str(from_node),
                    "target": str(to_node),
                    "value": 'Rch' + str(reach)}
        # Append the dictionary to the list
        links.append(dict_entry)

    # Print the list of dictionaries
    # print(f'Print the list of dictionaries {links}')

    json_data = {}
    json_data["nodes"] = nodes
    json_data["links"] = links

    verts = [] #can use a Python set here
    for v in json_data['nodes']:
        #print(f"{v['id']}")
        verts.append(v['id'])
    # print(f"{verts}\n")

    # Create an instance of the myGraph class
    my_graph = myGraph()

    # Add vertices the graph
    for v in json_data['nodes']:
        my_graph.add_vertex(v['id'])

    # Add edges to the graph
    for link in json_data['links']:
        if link['source'] in verts:
            my_graph.add_edge(link['source'], link['target'])
        else:
            print('New node')

    data = []
    for node in json_data['nodes']:
        SUB = node['id']
        if SUB == '0':
            continue
        ingoing = ' '.join(my_graph.incoming_nodes(SUB))
        outgoing = ' '.join(my_graph.outgoing_nodes(SUB))
        if '0' in my_graph.outgoing_nodes(SUB):
            outgoing = ''
        data.append({'REACH': SUB, 'Ingoing': ingoing, 'Outgoing': outgoing})

    # Create a DataFrame from the extracted data
    Final_Output_df = pd.DataFrame(data, columns=['REACH', 'Ingoing', 'Outgoing'])

    # Print the DataFrame
    #print(Final_Output_df)

    num_reaches = len(Final_Output_df)

    ############################# 4. Compute TP, TN Loads #########################
    Watershed_annual_df = pd.read_excel(subbasin_TN_input_file)
    Watershed_annual_df.rename(columns={'SUB': 'REACH'}, inplace=True)
    cols_to_group_by = ['REACH', 'YEAR']

    # TP
    cols_to_summarize_TP = ['ORGP_OUTkg', 'MINP_OUTkg']
    Watershed_annual_TP_df = Watershed_annual_df.groupby(cols_to_group_by)[cols_to_summarize_TP].sum()
    Watershed_annual_TP_df['TPtons'] = ( Watershed_annual_TP_df['ORGP_OUTkg'] + Watershed_annual_TP_df['MINP_OUTkg'] )/1000


    Watershed_annual_TP_df = Watershed_annual_TP_df.reset_index()

    df_tp_unstacked = Watershed_annual_TP_df.pivot(index='REACH', 
                                                columns='YEAR', 
                                                values='TPtons')

    # create Watershed_w_Split_annual_TN_df by using the outgoing reach
    Watershed_w_Split_annual_TP_df = pd.DataFrame()
    n_yrs = len(df_tp_unstacked.columns)
    Years = np.zeros(n_yrs)
    for i in range(n_yrs):
        Years[i] = df_tp_unstacked.columns[i]#avoid the for loop.
    Years = Years.astype(int)
    Fst_Yr = Years[0]
    Lst_Yr = Years[-1]
    for i in range(len(Final_Output_df)):
        TP_Load_tons = np.zeros(len(Years))
        # print('%s_%s' % (Final_Output_df_copy['REACH'].iloc[i], Final_Output_df_copy['Outgoing'].iloc[i]))
        for j in range(len(df_tp_unstacked)):
            if Final_Output_df['REACH'].iloc[i] == str(df_tp_unstacked.index[j]):
                # print('Reach =', Final_Output_df['REACH'].iloc[i], 'Outgoing = ' , Final_Output_df['Outgoing'].iloc[i])
                for y in Years:
                    TP_Load_tons[y - Fst_Yr] = df_tp_unstacked[y].iloc[i]
                Watershed_w_Split_annual_TP_df['%s_%s' % (Final_Output_df['REACH'].iloc[i], Final_Output_df['Outgoing'].iloc[i])] = TP_Load_tons 
                
    Watershed_w_Split_annual_TP_df = Watershed_w_Split_annual_TP_df.T

    # TN
    cols_to_summarize_TN = ['ORGN_OUTkg', 'NO3_OUTkg', 'NH4_OUTkg', 'NO2_OUTkg']
    Watershed_annual_TN_df = Watershed_annual_df.groupby(cols_to_group_by)[cols_to_summarize_TN].sum()
    Watershed_annual_TN_df['TNtons'] = ( Watershed_annual_TN_df['ORGN_OUTkg'] 
                                        + Watershed_annual_TN_df['NO3_OUTkg'] 
                                    + Watershed_annual_TN_df['NH4_OUTkg'] 
                                    + Watershed_annual_TN_df['NO2_OUTkg'] )/1000

    Watershed_annual_TN_df = Watershed_annual_TN_df.reset_index()

    df_tn_unstacked = Watershed_annual_TN_df.pivot(index='REACH', 
                                                columns='YEAR', 
                                                values='TNtons')

    # create Watershed_w_Split_annual_TN_df by using the outgoing reach
    Watershed_w_Split_annual_TN_df = pd.DataFrame()
    n_yrs = len(df_tn_unstacked.columns)
    Years = np.zeros(n_yrs)
    for i in range(n_yrs):
        Years[i] = df_tn_unstacked.columns[i]#avoid the for loop.
    Years = Years.astype(int)
    Fst_Yr = Years[0]
    Lst_Yr = Years[-1]
    for i in range(len(Final_Output_df)):
        TN_Load_tons = np.zeros(len(Years))
        # print('%s_%s' % (Final_Output_df_copy['REACH'].iloc[i], Final_Output_df_copy['Outgoing'].iloc[i]))
        for j in range(len(df_tn_unstacked)):
            if Final_Output_df['REACH'].iloc[i] == str(df_tn_unstacked.index[j]):
                # print('Reach =', Final_Output_df['REACH'].iloc[i], 'Outgoing = ' , Final_Output_df['Outgoing'].iloc[i])
                for y in Years:
                    TN_Load_tons[y - Fst_Yr] = df_tn_unstacked[y].iloc[i]
                Watershed_w_Split_annual_TN_df['%s_%s' % (Final_Output_df['REACH'].iloc[i], Final_Output_df['Outgoing'].iloc[i])] = TN_Load_tons 
                
    Watershed_w_Split_annual_TN_df = Watershed_w_Split_annual_TN_df.T


    # #################### 5. Compute TP, TN of the most dominant land usage.
    ### process ###########

    #process input_annual_subbasin_filename
    input_annual_subbasin_filename = os.path.join(Inputs_path, 
                                               input_annual_subbasin_filename)

    df = pd.read_excel(input_annual_subbasin_filename)

    # Drop YEAR and MON since you want averages across them
    df_no_time = df.drop(columns=["YEAR", "MON"])

    # Define grouping keys (non-numeric identifiers)
    group_cols = ["LULC", "HRU", "HRUGIS", "SUB"]

    # Take only numeric columns excluding group_cols
    numeric_cols = [col for col in df_no_time.select_dtypes(include="number").columns if col not in group_cols]

    # Group and average
    summary = df_no_time.groupby(group_cols)[numeric_cols].mean().reset_index()
    # # Save summarized result
    # input_average_subbasin_filename = 'SWAT_HRU_summary.xlsx'
    # SWAT_HRU_summary_input_file = os.path.join(Inputs_path, 
    #                                            input_average_subbasin_filename)
    # summary.to_excel(SWAT_HRU_summary_input_file, index=False)

    # input_annual_subbasin_filename = input_average_subbasin_filename

    # subbasin_input_file = os.path.join(Inputs_path, input_annual_subbasin_filename)

    # Watershed_annual_subbasin_df = pd.read_excel(subbasin_input_file)
    Watershed_annual_subbasin_df = summary

    # Watershed_annual_subbasin_df.info()

    # Watershed_annual_subbasin_df.info()

    # COLUMNS_TO_DROP = ['Area_ha', 'TP_tons', 'sum_percent_TP_TN']
    COLUMNS_TO_DROP = ['Area_ha', 'TP_tons', 'TN_tons', 'sum_percent_TP_TN']
    

    # COLUMNS_TO_KEEP = ['REACH', 'LUID', 'Area_acres', 
    #                    'percent_TP_tons_by_REACH']
    COLUMNS_TO_KEEP_TP = ['REACH', 'LUID', 'Area_acres', 'percent_TP_tons_by_REACH']
    COLUMNS_TO_KEEP_TN = ['REACH', 'LUID', 'Area_acres', 'percent_TN_tons_by_REACH']
    

    #out5_TP_file = 'Watershed_single_obj_opti_TP.csv'
    #out5_TP = os.path.join(Outputs_path, out5_TP_file)

    out5_TN_file = 'Watershed_single_obj_opti_TN.csv'
    out5_TN = os.path.join(Outputs_path, out5_TN_file)

    #out5_TP_TN_file = 'Watershed_multiple_obj_opti.csv'
    #out5_TP_TN = os.path.join(Outputs_path, out5_TP_TN_file)


    #calculate the Area_ac
    Watershed_annual_subbasin_df['Area_ac'] = Watershed_annual_subbasin_df['AREAkm2'] * 247.105
    # Watershed_annual_subbasin_df['Area_ac'] = ( 
    #         Watershed_annual_subbasin_df['Area_ac'].astype(int)
    # )

    Watershed_annual_subbasin_df['Area_ha'] = Watershed_annual_subbasin_df['AREAkm2'] * 100

    # # TP_tons = ORGPkg_ha + SEDPkg_h + SOLPkg_ha
    Watershed_annual_subbasin_df['TP_tons'] = Watershed_annual_subbasin_df['Area_ha'] * (Watershed_annual_subbasin_df['ORGPkg_ha'] + 
                                                                Watershed_annual_subbasin_df['SEDPkg_h'] + 
                                                                Watershed_annual_subbasin_df['SOLPkg_ha']) / 1000

    Watershed_annual_subbasin_df['TN_tons'] = Watershed_annual_subbasin_df['Area_ha'] * (Watershed_annual_subbasin_df['ORGNkg_ha'] + 
                                                                Watershed_annual_subbasin_df['NSURQkg_ha'] + 
                                                                Watershed_annual_subbasin_df['NO3GWkg_ha']) / 1000


    if 'SUB' in Watershed_annual_subbasin_df.columns:
        Watershed_annual_subbasin_df.rename(columns={'SUB': 'REACH'}, inplace=True)    


    if 'HRU' in Watershed_annual_subbasin_df.columns:
        Watershed_annual_subbasin_df.rename(columns={'LULC': 'LUID'}, inplace=True)


    cols_to_group_by = ['REACH', 'LUID'] 

    cols_to_summarize = ['Area_ha', 'Area_ac', 'TP_tons', 'TN_tons']

    # # check all columns in cols_to_summarize exist in Reach_k_annual_df
    for col in cols_to_summarize:
        if col not in Watershed_annual_subbasin_df.columns:
            print(f"Error: Column '{col}' does not exist in Watershed_annual_subbasin_df!")
            sys.exit("Exiting program.")

    # # Pivot table
    pivotT = Watershed_annual_subbasin_df.pivot_table(values=cols_to_summarize, 
                                            index=cols_to_group_by,
                                        aggfunc=sum)

    pivotT['percent_TP_tons_by_REACH'] = (pivotT['TP_tons'])/(pivotT.groupby(level='REACH')['TP_tons'].transform(sum))

    pivotT['percent_TP_tons_by_REACH'] = pivotT['percent_TP_tons_by_REACH'].replace(np.nan, 0, inplace=False)

    pivotT['percent_TN_tons_by_REACH'] = (pivotT['TN_tons'])/(pivotT.groupby(level='REACH')['TN_tons'].transform(sum))

    pivotT['percent_TN_tons_by_REACH'] = pivotT['percent_TN_tons_by_REACH'].replace(np.nan, 0, inplace=False)

    #pivotT['sum_percent_TP_TN'] = pivotT['percent_TP_tons_by_REACH'] + pivotT['percent_TN_tons_by_REACH']

    # TN
    pivotT['sum_percent_TP_TN'] = pivotT['percent_TN_tons_by_REACH']
    hru_df_single_obj_opti_TN = process_single_multi_obj_data(pivotT, 
                                'TN',
                                'single_obj', 
                                COLUMNS_TO_DROP, 
                                COLUMNS_TO_KEEP_TN, 
                                out5_TN)
    
    # # # 6. Collect data for the final outputs
    IP_Reaches_In_Out =  Final_Output_df
    IP_TP_Splitting = Watershed_w_Split_annual_TP_df
    IP_TN_Splitting = Watershed_w_Split_annual_TN_df

    # print("############################")
    # print(IP_TP_Splitting.head())
    #Create TP_df
    TP_df = pd.DataFrame()
    for i in range(len(IP_Reaches_In_Out)):
        TP_Out = np.zeros(len(Years))
        TP_In = np.zeros(len(Years))
        for reach_reach, cols in IP_TP_Splitting.iterrows():
            if IP_Reaches_In_Out['REACH'].iloc[i] == reach_reach.split("_")[0]:
                for y in Years:
                    TP_Out[y - Fst_Yr] = TP_Out[y - Fst_Yr] + cols[y- Fst_Yr]
            
            if IP_Reaches_In_Out['REACH'].iloc[i] == reach_reach.split("_")[1]:
                for y in Years:
                    TP_In[y - Fst_Yr] = TP_In[y - Fst_Yr] + cols[y- Fst_Yr]
                
        TP_df['%s' % (IP_Reaches_In_Out['REACH'].iloc[i])] = TP_Out - TP_In
    
    

    Final_Network_TP_df = process_new_tp_tn_df(TP_df, 
                                            IP_Reaches_In_Out, 
                                            Fst_Yr, 
                                            Lst_Yr)
    
    
    for i in range(len(IP_Reaches_In_Out)):
        TN_Out = np.zeros(len(Years))
        TN_In = np.zeros(len(Years))
        reach_id = str(IP_Reaches_In_Out['REACH'].iloc[i])
        found_out = False
        found_in = False
        for reach_reach, cols in IP_TN_Splitting.iterrows():
            out_id = reach_reach.split("_")[0]
            in_id = reach_reach.split("_")[1]
            if reach_id == out_id:
                found_out = True
            if reach_id == in_id:
                found_in = True
        print(f"REACH {reach_id}: found_out={found_out}, found_in={found_in}")
    
    #Create TN_df
    TN_df = pd.DataFrame()
    for i in range(len(IP_Reaches_In_Out)):
        TN_Out = np.zeros(len(Years))
        TN_In = np.zeros(len(Years))
        reach_id = str(IP_Reaches_In_Out['REACH'].iloc[i])
        for reach_reach, cols in IP_TN_Splitting.iterrows():
            if reach_id == reach_reach.split("_")[0]:
                for y in Years:
                    # print(f"REACH {reach_id}: cols[{y} - {Fst_Yr}]={cols[y - Fst_Yr]}")
                    TN_Out[y - Fst_Yr] = TN_Out[y - Fst_Yr] + cols[y- Fst_Yr]
            
            if reach_id == reach_reach.split("_")[1]:
                for y in Years:
                    TN_In[y - Fst_Yr] = TN_In[y - Fst_Yr] + cols[y- Fst_Yr]
                
        TN_df['%s' % (IP_Reaches_In_Out['REACH'].iloc[i])] = TN_Out - TN_In
        # print(f"REACH {reach_id}: TN_Out={TN_Out}, TN_In={TN_In}, Difference={TN_Out - TN_In}")


    Final_Network_TN_df = process_new_tp_tn_df(TN_df, 
                                            IP_Reaches_In_Out, 
                                            Fst_Yr, 
                                            Lst_Yr)

    ################ Compute splitting ratio ###################
    Splitting_TP_df = create_splitting_tp_tn_df(IP_Reaches_In_Out)
    Splitting_TN_df = create_splitting_tp_tn_df(IP_Reaches_In_Out)

    #Update Splitting_TP_df
    for i in range(len(IP_Reaches_In_Out)): # for each round in IP_Reaches_In_Out
        # if the current row's 'Outgoing' is a string variable then
        if type(IP_Reaches_In_Out['Outgoing'].iloc[i]) == str:
            # if there is more than 1 outgoing, then
            if len(IP_Reaches_In_Out['Outgoing'].iloc[i].split(",")) > 1:
                Splt_TN = [] # this variable is going to be changed.
                Splt_TP = [] # this variable is going to be changed.
                for j in IP_Reaches_In_Out['Outgoing'].iloc[i].split(","): # for each outgoing point then
                    try:
                        #Splt_TP.append(IP_TP_Splitting.loc['%s_%s'%(IP_Reaches_In_Out['REACH'].iloc[i],j)][:].mean())
                        Splt_TN.append(IP_TN_Splitting.loc['%s_%s'%(IP_Reaches_In_Out['REACH'].iloc[i],j)][:].mean())
                        Splt_TP.append(IP_TP_Splitting.loc['%s_%s'%(IP_Reaches_In_Out['REACH'].iloc[i],j)][:].mean())
                    except KeyError as e:
                        print(f"Error: Could not locate the row with key '{e.args[0]}' in IP_TP_Splitting.")
                        continue
                # for each element in Splt, if the element is negative, then remove it.
                #New_Splt_TP = [k for k in Splt_TP if k > 0]       
                New_Splt_TP = []
                for k in Splt_TP:
                    if k < 0:
                        New_Splt_TP.append(0)
                    else:
                        New_Splt_TP.append(k)
                
                New_Splt_TN = []
                for k in Splt_TN:
                    if k < 0:
                        New_Splt_TN.append(0)
                    else:
                        New_Splt_TN.append(k)
                Splitting_TN_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = ' '.join((k/sum(New_Splt_TN)).astype(str) for k in New_Splt_TN)
                Splitting_TP_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = ' '.join((k/sum(New_Splt_TP)).astype(str) for k in New_Splt_TP)
            else:
                #In Ratio column, If no multiple outgoings, then it was empty.
                Splitting_TP_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = ''
                Splitting_TN_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = ''

    Final_Network_TP_df = pd.merge(Final_Network_TP_df,
                                Splitting_TP_df, 
                                how = 'inner',
                                on = 'REACH')

    Final_Network_TN_df = pd.merge(Final_Network_TN_df,
                                Splitting_TN_df, 
                                how = 'inner',
                                on = 'REACH')

    ##############################################################################
    if time_periods is not None:
        Years = []
        for i in range(Fst_Yr, Lst_Yr+1):
            if str(i) in time_periods:
                Years.append(int(i))
        # if the Years is empty, then assign the range of years to Years
        if len(Years) == 0:
            # inform the time_periods is not in the range of years
            print(f"Warning: The time period '{time_periods}' is not in the input data!")
            # assign the range of years to Years
            Years = [int(i) for i in range(Fst_Yr, Lst_Yr+1)]
    else:
        Years = [int(i) for i in Years]
    # print(type(Years))
    final_columns_format_TN = ['REACH', 'Ingoing', 'Outgoing', 'Ratio'] + Years + ['LUID', 'Area_acres', 'percent_TN_tons_by_REACH']
    merged_df_single_obj_optim_TN = merge_ratio_TP_TN_percentage_data(Final_Network_TP_df, 
                                                                    hru_df_single_obj_opti_TN,
                                                                    final_columns_format_TN)


    COLUMNS_TO_DROP_TN = ['Ingoing', 'Outgoing', 'Ratio'] 
    Final_Network_TN_df.drop(columns=COLUMNS_TO_DROP_TN, inplace=True)

    

    Years_x = [str(i) + '_x' for i in Years]
    Years_y = [str(i) + '_y' for i in Years]

    final_columns_format_TP = ['REACH', 'Ingoing', 'Outgoing', 'Ratio'] + Years_x + Years_y + ['LUID', 'Area_acres', 'percent_TN_tons_by_REACH']
    merged_df_single_obj_optim_TN =  merge_ratio_TP_TN_percentage_data(merged_df_single_obj_optim_TN,
                                                                    Final_Network_TN_df,    
                                                                       final_columns_format_TP)
    
    
    ###################################################################################

    # if type(merged_df_single_obj_optim_TN['Area_acres']) != int:
    #     merged_df_single_obj_optim_TN['Area_acres'] = merged_df_single_obj_optim_TN['Area_acres'].astype(int)

    luid_df = load_data(input_luid_file)
    merged_df_single_obj_optim_TN = pd.merge(merged_df_single_obj_optim_TN,
                                            luid_df,
                                            how='left',
                                            left_on='LUID',
                                            right_on='swat_code')

    
    Years_x = [str(i) + '_x' for i in Years]
    Years_y = [str(i) + '_y' for i in Years]

    final_columns_format_TN = ['REACH', 'Ingoing', 'Outgoing', 'Ratio'] + Years_x + Years_y + ['WAM_LUID', 'Area_acres', 'percent_TN_tons_by_REACH']
    merged_df_single_obj_optim_TN = merged_df_single_obj_optim_TN[final_columns_format_TN]

    #rename the 'landuse_id' column to 'LUID'
    merged_df_single_obj_optim_TN.rename(columns={'WAM_LUID': 'LUID'}, inplace=True)

    #if type of LUID is not integer, LUID is integer
    merged_df_single_obj_optim_TN['LUID'] = merged_df_single_obj_optim_TN['LUID'].fillna(99999)
    if type(merged_df_single_obj_optim_TN['LUID']) != int:
        merged_df_single_obj_optim_TN['LUID'] = merged_df_single_obj_optim_TN['LUID'].astype(int)

    
    # process the merged_df_single_obj_optim_TN dataframe using the function, process_target_reaches under SWAT_utils.py
    merged_df_single_obj_optim_TN = process_target_reaches(merged_df_single_obj_optim_TN,
                                                           final_columns_format_TN,
                                                           Years_x= Years_x,
                                                              Years_y= Years_y)

    # round up Area to the nearest integer
    merged_df_single_obj_optim_TN['Area_acres'] = merged_df_single_obj_optim_TN['Area_acres'].fillna(1).apply(np.ceil).astype(int)

    # check Area_acres column for near zero value, assign 1 to it
    for i in range(len(merged_df_single_obj_optim_TN)):
        if merged_df_single_obj_optim_TN['Area_acres'].iloc[i] < 1 and merged_df_single_obj_optim_TN['Outgoing'].iloc[i] != '':
            print(f"Warning: Area_acres for REACH {merged_df_single_obj_optim_TN['REACH'].iloc[i]} is less than 1. Setting it to 1.")
            merged_df_single_obj_optim_TN['Area_acres'].iloc[i] = 1

    # Ensure no NaN values in merged_df_single_obj_optim_TN's percent_TN_tons_by_REACH column
    merged_df_single_obj_optim_TN['percent_TN_tons_by_REACH'] = merged_df_single_obj_optim_TN['percent_TN_tons_by_REACH'].fillna(0.0001)
    
    final_out_file_TN = 'SWAT_final_output_single_obj_optim_TN.csv'
    final_out_file_TN = os.path.join(Outputs_path, final_out_file_TN)
    merged_df_single_obj_optim_TN.to_csv(final_out_file_TN, index=False, header=True)

    ###################################################################################
    unique_LUID_TN = merged_df_single_obj_optim_TN['LUID'].unique()
    # remove LUID that is empty
    unique_LUID_TN = [luid for luid in unique_LUID_TN if luid != '']
    unique_LUID_TN = np.array(unique_LUID_TN, dtype=int)
    # create a new dataframe given by unique_LUID_TN and data type is integer
    unique_LUID_TN = pd.DataFrame(unique_LUID_TN, columns=['LUID']) 


    # save the numpy array to a CSV file
    final_out_file_TN = 'SWAT_unique_LUID_optim_TN.csv'
    final_out_file_TN = os.path.join(Outputs_path, final_out_file_TN)
    unique_LUID_TN.to_csv(final_out_file_TN, index=False, header=True)

if __name__ == "__main__":
    #srun --nodes=1 --partition=general --pty /bin/bash
    # test with small inputs.
    if len(sys.argv) == 1:
        print("Usage: python SWAT_Network_Automation_TN.py <time_periods> <working_path>")
        print("Example: python SWAT_Network_Automation_TN.py '2018, 2020' /path/to/working_directory")
        time_periods = None
        working_path = os.getcwd()
        swat_network_automation_tn(working_path, time_periods)
        # sys.exit(1)

    else:
        time_periods = sys.argv[1]
        working_path = sys.argv[2]
        swat_network_automation_tn(working_path, time_periods)
