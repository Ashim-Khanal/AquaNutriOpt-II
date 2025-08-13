# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:09:44 2024

@author: osama
"""

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
from AquaNutriOpt.utils import *


def wam_network_automation_mo(working_path: str, time_periods: str):
    """Performs network automation for multi-objective optimization on WAM inputs.
    
    Args:
        working_path (str): The path to the working directory where WAM inputs and outputs are stored.
        time_periods (str): Time periods to process (e.g., "2018", "2018, 2020").
    """
    
    # Example usage

    Wam_path = os.path.join(working_path, 'WAM')
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


    data_dir = 'WAM/Inputs/Reaches' #Directory where all the WAM's outputs, reach *.csv files, are stored
    input_TP_filename = 'WAM/Outputs/Watershed_Subbasin_LU_TP_MO.xlsx'
    input_TN_filename = 'WAM/Outputs/Watershed_Subbasin_LU_TN_MO.xlsx'
    subbasin_TP_input_file = os.path.join(working_path, input_TP_filename)
    subbasin_TN_input_file = os.path.join(working_path, input_TN_filename)

    #check if the data_dir and data_dir_2 exist in the Working_path. Otherwise, print an error message and exit from the program.
    if not os.path.exists(os.path.join(working_path, data_dir)):
        print(f"Error: Directory '{data_dir}' does not exist in the current working directory!")
        sys.exit("Exiting program.")

    input_data_files = os.path.join(working_path, data_dir)

    # create a sub-folder under the Working_path
    if not os.path.exists(os.path.join(working_path, 'WAM')):
        os.makedirs(os.path.join(working_path, 'WAM'))

    out1_file = 'WAM/Outputs/Watershed_Annual_Flow.csv'
    out1 = os.path.join(working_path, out1_file)

    out2_file = 'WAM/Outputs/Watershed_Reaches_In_Out.csv'
    out2 = os.path.join(working_path, out2_file)

    #Collect all reaches data
    All_reaches = [f for f in listdir(input_data_files) if isfile(join(input_data_files, f))]
      
    Watershed_daily_Flow_df, num_days = gen_reach_j_daily_df(input_data_files, All_reaches)

    # print(f'Reach_j_daily_df')
    # print(Reach_j_daily_df.head())
    # print(Reach_j_daily_df.tail())
    # print(Reach_j_daily_df.info())


    # # Populate Reach_j_daily_dfReach_j_daily_df, Reach_IDs and DS_Reach
    Reach_IDs = [] # a list of upstream reaches
    DS_Reach = [] # a list of downstream (DS) reachers
    Watershed_daily_Flow_df, Reach_IDs, DS_Reach = populate_Reach_IDs_DS_Reach(input_data_files, 
                                                    All_reaches, 
                                                    Watershed_daily_Flow_df,
                                                    Reach_IDs, 
                                                    DS_Reach, 
                                                    num_days)   


    # Watershed_annual_Flow_df = calculate_sum_annual_flow_vol_and_save(Watershed_daily_Flow_df, 
    #                                                                     out1)

    
    ########### Represent a graph of reaches and search the graph for the Ingoing and Outgoing reaches ###################
    # # Update Reach_IDs_DS_df
    # ## Identify Downstream Reaches

    Reach_IDs_DS_df = pd.DataFrame() # we use a Pandas data frame to store Reaches Downstream per reach
    Reach_IDs_DS_df['Reach_ID'] = Reach_IDs
    Reach_IDs_DS_df['DS_Reach'] = DS_Reach

    #Identify every unique Reach through
    #appending all Reaches from Reach and DS Reach and 
    #then remove the duplicates!
    # Reaches = Reach_IDs_DS_df['Reach_ID']
    # Reaches = Reaches.append(Reach_IDs_DS_df['DS_Reach'])

    Reaches = pd.concat([Reach_IDs_DS_df['Reach_ID'],Reach_IDs_DS_df['DS_Reach']])
    Reaches_df = pd.DataFrame(Reaches, columns = ['Reaches'])
    Reaches_df = Reaches_df.drop_duplicates(subset=['Reaches'])

    #Number of unique Reaches.
    Reaches_num = len(Reaches_df.index)
    #Number of Reach_IDs
    Total_num = len(Reach_IDs_DS_df.index)

    # Ingoing_Reaches = []
    # Outgoing_Reaches = []
    Ingoing_Reaches_df = pd.DataFrame()
    Outgoing_Reaches_df = pd.DataFrame()

    # Here I will specify Reach and look for it in the DS_Reach column and 
    # then identify the associated Nodes in the Reach Column
    # Thus, I will identify all the Reaches that flow into the identified Reach.
    for i in range(Reaches_num):
        Ingoing_array = []
        for j in range(Total_num):
            if Reaches_df['Reaches'].iloc[i] == Reach_IDs_DS_df['DS_Reach'].iloc[j]:
                Ingoing = Reach_IDs_DS_df['Reach_ID'].iloc[j]
            else:
                Ingoing = np.nan
            Ingoing_array.append(Ingoing)
        Ingoing_Reaches_df['%s'%Reaches_df['Reaches'].iloc[i]] = Ingoing_array
        Ingoing_Reaches_df['%s'%Reaches_df['Reaches'].iloc[i]] = Ingoing_Reaches_df['%s'%Reaches_df['Reaches'].iloc[i]].drop_duplicates()
    Ingoing_Reaches_Filtered_df = Ingoing_Reaches_df.apply(lambda x: ' '.join(x.astype(str).replace('\.0',"",regex=True)))

    N = len(Ingoing_Reaches_Filtered_df.index)
    for j in range(N):
        Ingoing_Reaches_Filtered_df.iloc[j] = ' '.join([ i for i in Ingoing_Reaches_Filtered_df.iloc[j].split(' ') if i != 'nan' ])

    # Here I will specify Reach and look for it in the Reach column and 
    # then identify the associated Reaches in the DS_Reach column
    # Thus, I will identify all the Reaches that receive flow from the identified Reach.
    for i in range(Reaches_num):
        Outgoing_array = []
        for j in range(Total_num):
            if Reaches_df['Reaches'].iloc[i] == Reach_IDs_DS_df['Reach_ID'].iloc[j]:
                Outgoing = Reach_IDs_DS_df['DS_Reach'].iloc[j]
            else:
                Outgoing = np.nan
            Outgoing_array.append(Outgoing)
        Outgoing_Reaches_df['%s'%Reaches_df['Reaches'].iloc[i]] = Outgoing_array
        Outgoing_Reaches_df['%s'%Reaches_df['Reaches'].iloc[i]] = Outgoing_Reaches_df['%s'%Reaches_df['Reaches'].iloc[i]].drop_duplicates()
    Outgoing_Reaches_Filtered_df = Outgoing_Reaches_df.apply(lambda x: ' '.join(x.astype(str).replace('\.0',"",regex=True)))
    N = len(Outgoing_Reaches_Filtered_df.index)
    for j in range(N):
        Outgoing_Reaches_Filtered_df.iloc[j] = ' '.join([ i for i in Outgoing_Reaches_Filtered_df.iloc[j].split(' ') if i != 'nan' ])

    Final_Output_df = pd.DataFrame()
    Final_Output_df['REACH'] = Reaches_df['Reaches'].values
    Final_Output_df['Ingoing'] = Ingoing_Reaches_Filtered_df.values
    Final_Output_df['Outgoing'] = Outgoing_Reaches_Filtered_df.values
    # Final_Output_df.to_csv(out2, index=False, header=True)  

    ##################Compute TP, TN Loads ######################################################################

    out3_TP_file = 'Watershed_Base_Annual_TP_new.csv'
    out3_TP = os.path.join(Outputs_path, out3_TP_file)

    out3_TN_file = 'Watershed_Base_Annual_TN_new.csv'
    out3_TN = os.path.join(Outputs_path, out3_TN_file)

    out4_TP_file = 'Watershed_Base_Annual_TP_w_Split_new.csv'
    out4_TP = os.path.join(Outputs_path, out4_TP_file)

    out4_TN_file = 'Watershed_Base_Annual_TN_w_Split_new.csv'
    out4_TN = os.path.join(Outputs_path, out4_TN_file)

    #Read data of all reaches
    All_reaches = [f for f in listdir(input_data_files) if isfile(join(input_data_files, f))]

    num_reaches = len(All_reaches)

    myNaNCols = ['SedPIn','SolPIn', 'SedPOut', 'SolPOut', 'SolNO3Out', \
            'SolNH4Out', 'SolOrgNOut', 'SedNH4Out', 'SedOrgNOut', 'SolNO3In', \
            'SolNH4In', 'SolOrgNIn', 'SedNH4In', 'SedOrgNIn', 'FlowOut', \
            'FlowOutFraction', 'FlowIn', 'FlowInFraction' 
            ]

    Watershed_w_Split_daily_TP_df, num_days = gen_reach_j_daily_df(input_data_files, 
                                                            All_reaches)

    Watershed_w_Split_daily_TN_df, num_days = gen_reach_j_daily_df(input_data_files, 
                                                            All_reaches)

    # ### Update Watershed_daily_df
    Watershed_daily_TP_TN_df, num_days = gen_reach_j_daily_df(input_data_files, 
                                                    All_reaches)
    # # The For-Loop

    # ### Update Watershed_w_Split_daily_df
    Watershed_w_Split_daily_TP_df, Watershed_w_Split_daily_TN_df, Watershed_daily_TP_TN_df = computeReach_TP_TN_Loads(input_data_files,
                                                                                        All_reaches,
                                                                                        Watershed_w_Split_daily_TP_df,
                                                                                        Watershed_w_Split_daily_TN_df,
                                                                                        Watershed_daily_TP_TN_df,
                                                                                        num_days,
                                                                                        myNaNCols)

    Watershed_w_Split_annual_TP_df = downsamplingDF(Watershed_w_Split_daily_TP_df)
    # Watershed_w_Split_annual_TP_df.to_csv(out4_TP, 
    #                                       index=True, 
    #                                       header=True)

    Watershed_w_Split_annual_TN_df = downsamplingDF(Watershed_w_Split_daily_TN_df)
    # Watershed_w_Split_annual_TN_df.to_csv(out4_TN, index=True, header=True)



    Watershed_annual_TP_df = downsamplingDF(Watershed_daily_TP_TN_df)
    Watershed_annual_TP_df = Watershed_annual_TP_df.T
    Watershed_annual_TP_df = Watershed_annual_TP_df.filter(like='TP', axis=1)
    Watershed_annual_TP_df = Watershed_annual_TP_df.T
    # Watershed_annual_TP_df.to_csv(out3_TP, index=True, header=True)

    Watershed_annual_TN_df = downsamplingDF(Watershed_daily_TP_TN_df)
    Watershed_annual_TN_df = Watershed_annual_TN_df.T
    Watershed_annual_TN_df = Watershed_annual_TN_df.filter(like='TN', axis=1)
    Watershed_annual_TN_df = Watershed_annual_TN_df.T
    # Watershed_annual_TN_df.to_csv(out3_TN, index=True, header=True)

    #####################Compute the Reach_LUID with the max TP, TN, TP+TN loads###################################################################

    # COLUMNS_TO_DROP = ['Area_ha', 'TP_tons', 'sum_percent_TP_TN']
    COLUMNS_TO_DROP = ['Area_ha', 'TP_tons', 'TN_tons', 'sum_percent_TP_TN']
    COLUMNS_TO_DROP_TP_TN = ['Area_ha', 'TP_tons', 'TN_tons']

    # COLUMNS_TO_KEEP = ['REACH', 'LUID', 'Area_acres', 
    #                    'percent_TP_tons_by_REACH']
    COLUMNS_TO_KEEP_TP = ['REACH', 'LUID', 'Area_acres', 'percent_TP_tons_by_REACH']

    COLUMNS_TO_KEEP_TN = ['REACH', 'LUID', 'Area_acres', 'percent_TN_tons_by_REACH']

    COLUMNS_TO_KEEP_TP_TN = ['REACH', 'LUID', 'Area_acres', 'percent_TP_tons_by_REACH', 'percent_TN_tons_by_REACH', 'sum_percent_TP_TN']

    #out5_TP_file = 'Watershed_single_obj_opti_TP.csv'
    #out5_TP = os.path.join(Outputs_path, out5_TP_file)

    # out5_TN_file = 'Watershed_single_obj_opti_TN.csv'
    # out5_TN = os.path.join(Outputs_path, out5_TN_file)

    out6_file = 'Watershed_multiple_obj_opti.csv'
    out6 = os.path.join(Outputs_path, out6_file)

    Watershed_Subbasin_LU_TP = load_data(subbasin_TP_input_file)

    Watershed_Subbasin_LU_TP['Area_ac'] = Watershed_Subbasin_LU_TP['Area_ac'].astype(int)
    Watershed_Subbasin_LU_TP['Area_ha'] = Watershed_Subbasin_LU_TP['Area_ac'] * 0.404686
    Watershed_Subbasin_LU_TP['TP_tons'] = Watershed_Subbasin_LU_TP['Area_ha'] * Watershed_Subbasin_LU_TP['TP_kg/ha'] / 1000
    # Rename 'Reach' to 'REACH'
    # check if 'Reach' exists in Watershed_annual_df
    if 'Reach' in Watershed_Subbasin_LU_TP.columns:
        Watershed_Subbasin_LU_TP.rename(columns={'Reach': 'REACH'}, inplace=True)
    elif 'REACH' in Watershed_Subbasin_LU_TP.columns:
        pass
    else:
        print("Error: Column 'Reach' does not exists in Watershed_annual_df!")
        #exit the program
        sys.exit("Exiting program.")

    Watershed_Subbasin_LU_TN = load_data(subbasin_TN_input_file)
    # Rename Subbasin to REACH
    if 'Subbasin' in Watershed_Subbasin_LU_TN.columns:
        Watershed_Subbasin_LU_TN.rename(columns={'Subbasin': 'REACH'}, inplace=True)

    # Rename TN to TN_tons
    if 'TN' in Watershed_Subbasin_LU_TN.columns:
        Watershed_Subbasin_LU_TN.rename(columns={'TN': 'TN_kg/ha'}, inplace=True)


    Watershed_Subbasin_LU_TN['Area_ha'] = Watershed_Subbasin_LU_TN['Area_ft2'] /  107600
    Watershed_Subbasin_LU_TN['TN_tons'] = Watershed_Subbasin_LU_TN['Area_ha'] * Watershed_Subbasin_LU_TN['TN_kg/ha'] / 1000

    #drop Area_ha column
    Watershed_Subbasin_LU_TN.drop(columns=['Area_ha'], inplace=True)
    
    # Drop overlapping columns in Watershed_annual_TP_df and Watershed_annual_TN_df
    Watershed_Subbasin_LU_TN.drop(columns=['Area_ft2', 'Area_ac'], inplace=True)
    
    # merge the two dataframes
    Watershed_Subbasin_TP_TN_df = pd.merge(Watershed_Subbasin_LU_TP, 
                                                    Watershed_Subbasin_LU_TN, 
                            left_index=True,
                                       right_index=True,
                            how='left')

    desired_columns = ['REACH_x', 'LUID_y', 'Area_ha', 'Area_ac', 'TP_tons', 'TN_tons']
    Watershed_Subbasin_TP_TN_df = Watershed_Subbasin_TP_TN_df[desired_columns]

    Watershed_Subbasin_TP_TN_df = Watershed_Subbasin_TP_TN_df.rename(
        columns={'REACH_x': 'REACH', 'LUID_y': 'LUID'}
    )

    cols_to_group_by = ['REACH', 'LUID'] 

    cols_to_summarize = ['Area_ha', 'Area_ac', 'TP_tons', 'TN_tons']

    # check all columns in cols_to_summarize exist in Reach_k_annual_df
    for col in cols_to_summarize:
        if col not in Watershed_Subbasin_TP_TN_df.columns:
            print(f"Error: Column '{col}' does not exist in Reach_k_annual_df!")
            sys.exit("Exiting program.")

    # # Pivot table
    pivotT = Watershed_Subbasin_TP_TN_df.pivot_table(values=cols_to_summarize, 
                                            index=cols_to_group_by,
                                        aggfunc=sum)

    pivotT['percent_TP_tons_by_REACH'] = (pivotT['TP_tons'])/(pivotT.groupby(level='REACH')['TP_tons'].transform(sum))

    pivotT['percent_TP_tons_by_REACH'] = pivotT['percent_TP_tons_by_REACH'].replace(np.nan, 0, inplace=False)

    pivotT['percent_TN_tons_by_REACH'] = (pivotT['TN_tons'])/(pivotT.groupby(level='REACH')['TN_tons'].transform(sum))

    pivotT['percent_TN_tons_by_REACH'] = pivotT['percent_TN_tons_by_REACH'].replace(np.nan, 0, inplace=False)

    pivotT['sum_percent_TP_TN'] = pivotT['percent_TP_tons_by_REACH'] + pivotT['percent_TN_tons_by_REACH']
    # pivotT['sum_percent_TP_TN'] = pivotT['percent_TP_tons_by_REACH']

    ##############multiple objective optimization####################

    hru_df_multiple_obj_opti_TP_TN = process_single_multi_obj_data(pivotT, 
                                'TP_TN',
                                'multi_obj',
                                COLUMNS_TO_DROP_TP_TN, 
                                COLUMNS_TO_KEEP_TP_TN, 
                                out6)

    ################# Collect data for the final outputs ########################
    # # Load the data from the previously saved CSVs
    IP_Reaches_In_Out =  Final_Output_df
    IP_TP_Splitting = Watershed_w_Split_annual_TP_df
    IP_TN_Splitting = Watershed_w_Split_annual_TN_df
    # hru_df_single_obj_opti_TP = pd.read_csv(out5_TP)
    # hru_df_single_obj_opti_TN = pd.read_csv(out5_TN)
    # hru_df_multiple_obj_opti_TP_TN = pd.read_csv(out6)

    #This for loop determines TP loads 
    # generated within the Node's Subbasin 
    # by subtracting TP out of this Node
    # - TP into this Node.
    Fst_Yr = int(IP_TP_Splitting.columns[0])
    # print(Fst_Yr)
    Lst_Yr = int(IP_TP_Splitting.columns[-1])
    # print(Lst_Yr)
    Years = range(Fst_Yr, Lst_Yr+1)

    # create a new dataframe where the number of columns is equal to the number of years
    # each column is float64 data type
    # and number of rows is equal to the number of reaches

    TP_df = update_tp_tn_df(IP_Reaches_In_Out, IP_TP_Splitting, Fst_Yr, Lst_Yr) 
    TN_df = update_tp_tn_df(IP_Reaches_In_Out, IP_TN_Splitting, Fst_Yr, Lst_Yr) 

    Final_Network_TP_df = process_new_tp_tn_df(TP_df, 
                                            IP_Reaches_In_Out, 
                                            Fst_Yr, 
                                            Lst_Yr)

    Final_Network_TN_df = process_new_tp_tn_df(TN_df, 
                                            IP_Reaches_In_Out, 
                                            Fst_Yr, 
                                            Lst_Yr)
    ###################### Compute the Splitting Ratios based on TP ########################################
    Splitting_TP_df = create_splitting_tp_tn_df(IP_Reaches_In_Out)
    Splitting_TN_df = create_splitting_tp_tn_df(IP_Reaches_In_Out)

    for i in range(len(IP_Reaches_In_Out)): # for each round in IP_Reaches_In_Out
        # if the current row's 'Outgoing' is a string variable then
        if type(IP_Reaches_In_Out['Outgoing'].iloc[i]) == str:
            # if there is more than 1 outgoing, then
            if len(IP_Reaches_In_Out['Outgoing'].iloc[i].split(" ")) > 1:
                Splt_TP = [] # this variable is going to be changed.
                Splt_TN = [] # this variable is going to be changed.
                for j in IP_Reaches_In_Out['Outgoing'].iloc[i].split(" "): # for each outgoing point then
                    try:
                        Splt_TP.append(IP_TP_Splitting.loc['%s_%s'%(IP_Reaches_In_Out['REACH'].iloc[i],j)][:].mean())
                        Splt_TN.append(IP_TN_Splitting.loc['%s_%s'%(IP_Reaches_In_Out['REACH'].iloc[i],j)][:].mean())
                    except KeyError as e:
                        print(f"Error: Could not locate the row with key '{e.args[0]}' in IP_TP_Splitting.")
                        continue
                # for each element in Splt, if the element is negative, then remove it.
                New_Splt_TN = []
                for k in Splt_TN:
                    if k < 0:
                        New_Splt_TN.append(0)
                    else:
                        New_Splt_TN.append(k)
                
                New_Splt_TP = []
                for k in Splt_TP:
                    if k < 0:
                        New_Splt_TP.append(0)
                    else:
                        New_Splt_TP.append(k) 
                Splitting_TP_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = ' '.join((k/sum(New_Splt_TP)).astype(str) for k in New_Splt_TP)
                Splitting_TN_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = ' '.join((k/sum(New_Splt_TN)).astype(str) for k in New_Splt_TN)
            else:
                # Splitting_df['Reach'].loc[IP_Reaches_In_Out['Reach'].iloc[i]] = np.nan
                Splitting_TP_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = np.nan
                Splitting_TN_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = np.nan
        else:
            # Splitting_df['Reach'].loc[IP_Reaches_In_Out['Reach'].iloc[i]] = np.nan
            Splitting_TP_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = np.nan
            Splitting_TN_df['Ratio'].loc[IP_Reaches_In_Out['REACH'].iloc[i]] = np.nan

    # export the Splitting_df to a CSV file
    # Splitting_df.to_csv('./UK/UK_TP_Splitting.csv', index=True, header=True)

    #should we rename IP_Reaches_In_Out?
    Final_Network_TP_df = pd.merge(Final_Network_TP_df,
                                Splitting_TP_df, 
                                how = 'inner', 
                                on ='REACH')

    Final_Network_TN_df = pd.merge(Final_Network_TN_df,
                                Splitting_TN_df, 
                                how = 'inner', 
                                on ='REACH')

    ##############################################################################

    
    # merge with Final_Network_TP_df
    hru_df_multiple_obj_opti_TP_TN['REACH'] = hru_df_multiple_obj_opti_TP_TN['REACH'].astype('int64')

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
    
            
    final_columns_format_TP_TN = ['REACH', 'Ingoing', 'Outgoing', 'Ratio'] + Years + ['LUID', 'Area_acres', 'percent_TP_tons_by_REACH', 'percent_TN_tons_by_REACH']
    merged_df_multi_obj_optim = merge_ratio_TP_TN_percentage_data(  Final_Network_TP_df, 
                                                                    hru_df_multiple_obj_opti_TP_TN, 
                                                                    final_columns_format_TP_TN )


    #drop Final_Network_TN_df['Ratio'] column
    #
    COLUMNS_TO_DROP_TN = ['Ingoing', 'Outgoing', 'Ratio'] 
    Final_Network_TN_df.drop(columns=COLUMNS_TO_DROP_TN, inplace=True)
    #drop 

    Years_x = [str(i) + '_x' for i in Years]
    Years_y = [str(i) + '_y' for i in Years]
    final_columns_format_TP_TN = ['REACH', 'Ingoing', 'Outgoing', 'Ratio'] + Years_x + Years_y +  ['LUID', 'Area_acres', 'percent_TP_tons_by_REACH', 'percent_TN_tons_by_REACH']
    merged_df_multi_obj_optim = merge_ratio_TP_TN_percentage_data(merged_df_multi_obj_optim,
                                                                Final_Network_TN_df,
                                                                final_columns_format_TP_TN)

    merged_df_multi_obj_optim['LUID'] = merged_df_multi_obj_optim['LUID'].fillna(0)
    if type(merged_df_multi_obj_optim['LUID']) != int:
        merged_df_multi_obj_optim['LUID'] = merged_df_multi_obj_optim['LUID'].astype(int)


    # Process merged_df_multi_obj_optim using util.py's process_target_reaches method
    merged_df_multi_obj_optim = process_target_reaches(merged_df_multi_obj_optim,
                                                       Years_x=Years_x,
                                                       Years_y=Years_y)


    #####################
    # round up the 'Area_acres' in the dataframe to next whole number
    merged_df_multi_obj_optim['Area_acres'] = merged_df_multi_obj_optim['Area_acres'].fillna(0).apply(np.ceil).astype(int)
    
    # for each row in the merged_df_multi_obj_optim['REACH'] column,
    # if the area is less than 1 and the Outgoing is not empty, then set the area to 1
    for i in range(len(merged_df_multi_obj_optim)):
        if merged_df_multi_obj_optim['Area_acres'].iloc[i] < 1 and \
           merged_df_multi_obj_optim['Outgoing'].iloc[i] != '':
            print(f"Setting Area_acres to 1 for REACH {merged_df_multi_obj_optim['REACH'].iloc[i]}")
            merged_df_multi_obj_optim['Area_acres'].iloc[i] = 1

    ########################
    
    # rename 'percent_TP_tons_by_REACH' to 'TP_percent'
    merged_df_multi_obj_optim.rename(columns={'percent_TP_tons_by_REACH': 'TP_percent'}, inplace=True)

    # rename 'percent_TN_tons_by_REACH' to 'TN_percent'
    merged_df_multi_obj_optim.rename(columns={'percent_TN_tons_by_REACH': 'TN_percent'}, inplace=True)

    # # Export the final merged DataFrame to a CSV file
    final_out_file = 'WAM_final_output_multiple_obj_optim.csv'
    merged_df_multi_obj_optim.to_csv(os.path.join(Outputs_path, final_out_file), index=False, header=True)

    # extract unique values of merged_df_single_obj_optim_TN['LUID']
    unique_LUID = merged_df_multi_obj_optim['LUID'].unique()
    # remove '' values from unique_LUID_TN
    unique_LUID = [x for x in unique_LUID if x != '']
    unique_LUID = np.array(unique_LUID, dtype=int)
    unique_LUID = pd.DataFrame(unique_LUID, columns=['LUID'])

    final_out_file = 'WAM_unique_LUID_multiple_obj_optim.csv'
    unique_LUID.to_csv(os.path.join(Outputs_path, final_out_file), index=False, header=True)

    # print(f"Merged data has been saved to {final_out_file}")


if __name__ == "__main__":
    #srun --nodes=1 --partition=general --pty /bin/bash
    # test with small inputs.
    if len(sys.argv) == 1:
        print("Usage: python WAM_Network_Automation_MO.py <time_periods> <working_path>")
        print("Example: python WAM_Network_Automation_MO.py '2018, 2020' /path/to/working_directory")
        time_periods = None
        working_path = os.getcwd()
        wam_network_automation_mo(working_path, time_periods)
        # sys.exit(1)

    else:
        time_periods = sys.argv[1]
        working_path = sys.argv[2]
        wam_network_automation_mo(working_path, time_periods)
