import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import warnings
import csv

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def load_data(file):
    """
    Loads data from an Excel file and returns it as a DataFrame.

    Args:
        file (str): The path to the Excel file to be loaded.

    Returns:
        pandas.DataFrame: A copy of the first sheet of the Excel file as a DataFrame.
    """
    xls = pd.ExcelFile(file)
    df = xls.parse(0)
    return df.copy()

def gen_reach_j_daily_df(input_data_files, All_reaches):
    """
    Create a Pandas's daily (time) related Dataframe for a specific reach based on the first input file.
    Args:
        input_data_files (str): The directory path where the input data files are located.
        All_reaches (list): A list of filenames for all reaches.
    Returns:
        tuple: A tuple containing:
            - Reach_j_daily_df (pd.DataFrame): A dataframe with a date range for the specific reach.
            - num_days (int): The number of days in the reach data.
    """
    #All_reaches[0]
    # print(All_reaches[0])
    Reach_j = pd.read_csv(os.path.join(input_data_files, All_reaches[0]))
    Reach_j = Reach_j.iloc[1:, :] #ignore the first row
    Reach_j['SimDate'] = pd.to_datetime(Reach_j['SimDate'])  # convert it to a datetime object
    Reach_j['date'] = Reach_j['SimDate'].dt.date  # extract date info

    # Calculate number of rows (days) in each file
    num_days = len(Reach_j.index)

    # Create a pd.DateTimeIndex object to generate a sequence of dates
    index = pd.date_range(start=Reach_j['date'].iloc[0],
                                      end=Reach_j['date'].iloc[-1], 
                                      freq='D')

    # Generate a blank dataframe that will include all reaches daily data within the specified date range.
    Reach_j_daily_df = pd.DataFrame(index,
                                    columns=['date'])
    
    return Reach_j_daily_df, num_days

def populate_Reach_IDs_DS_Reach(input_data_files, All_reaches, Reach_j_daily_df, 
                                Reach_IDs, DS_Reach, num_days):
    """
    Populates the daily flow data for each reach and updates the lists of Reach IDs and downstream Reach IDs.
    Args:
        input_data_files (str): The directory path where the input data files are located.
        All_reaches (list): A list of filenames for each reach's data.
        Reach_j_daily_df (pd.DataFrame): A DataFrame to store the daily flow data for each reach.
        Reach_IDs (list): A list to store the Reach IDs.
        DS_Reach (list): A list to store the downstream Reach IDs.
        num_days (int): The number of days for which the flow data is to be calculated.
    Returns:
        pd.DataFrame: Updated DataFrame with daily flow data for each reach.
        list: Updated list of Reach IDs.
        list: Updated list of downstream Reach IDs.
    """
    num_reaches = len(All_reaches)
    for j in range(num_reaches):
        Reach_j = pd.read_csv(os.path.join(input_data_files, All_reaches[j]))
        Reach_j = Reach_j.iloc[1:, :]
        
        # Collect Flow data for each reach
        Flow_daily_m3_day = np.zeros(num_days)
        Flow_daily_m3_day = Reach_j['Flow'] * 60 * 60 * 24  # convert from m3/secons to m3/days
        
        # update Reach_j_daily_df with Flow data
        Reach_j_daily_df['%s_%s'%(Reach_j['ReachID'].iloc[0], Reach_j['ReachNextID'].iloc[0])] = Flow_daily_m3_day
        
        # update Reach_IDs
        Reach_IDs.append(Reach_j['ReachID'].iloc[0])

        # update DS_Reach
        DS_Reach.append(Reach_j['ReachNextID'].iloc[0])

    return Reach_j_daily_df, Reach_IDs, DS_Reach

def calculate_sum_annual_flow_vol_and_save(Reach_j_daily_df, out1):
    """
    Calculate the annual sum of flow volumes from daily data and save the result to a CSV file.
    Parameters:
    Reach_j_daily_df (pd.DataFrame): DataFrame containing daily flow data with a 'date' column.
    out1 (str): File path to save the resulting annual flow volume data.
    Returns:
    pd.DataFrame: Transposed DataFrame of annual flow volumes with years as index.
    """
    # Convert index to datetime
    Reach_j_daily_df = Reach_j_daily_df.set_index('date')
    
    Reach_j_daily_df.index = pd.to_datetime(Reach_j_daily_df.index, 
                                            unit='s')

    # Resample to monthly and then to annual sum
    Reach_j_monthly_df = Reach_j_daily_df.resample('M').sum()
    Reach_j_annual_df = Reach_j_monthly_df.resample('Y').sum()

    # Rename column headers to represent reach ID
    n_yrs = len(Reach_j_annual_df.index)
    Year = np.zeros(n_yrs, dtype=int)
    for i in range(n_yrs):
        Year[i] = Reach_j_annual_df.index[i].year

    Reach_j_annual_df['Year'] = Year
    Reach_j_annual_df = Reach_j_annual_df.set_index('Year')
    Reach_j_annual_Trans_df = Reach_j_annual_df.T.rename(columns={0: "Reach"})

    # Save the modified dataframe to a CSV file
    # Reach_j_annual_Trans_df.to_csv(out1, index=False, header=True)
    return Reach_j_annual_Trans_df


# ## handleNaNValues. In-place function
def handleNaNValues(Reach_j, col):
    """
    Handles NaN values in a specified column of a DataFrame.

    This function checks if the specified column exists in the DataFrame. 
    
    If it does, it replaces all NaN values in the column with 0 and converts the column to numeric, 
    coercing any non-numeric values to NaN. 
    
    If the column does not exist, it prints an error message and exits the program.

    Parameters:
    Reach_j (pd.DataFrame): The DataFrame to process.
    col (str): The name of the column to handle NaN values for.

    Returns:
    pd.DataFrame: The DataFrame with NaN values handled in the specified column.
    """
    if col in Reach_j.columns:
        # Reach_j[col] = Reach_j[col].replace(np.NaN, 0, inplace=True)
        Reach_j[col] = Reach_j[col].replace(np.nan, 0, inplace=False)
        Reach_j[col] = pd.to_numeric(Reach_j[col], errors='coerce') #'coerce' replace all non-numeric values with NaN.
    else: 
        print(f"Error: Column '{col}' does not exist!")
        sys.exit("Exiting program.")
    return Reach_j


# ## calculate_TP_Load_tons
def calculate_TP_Load_tons(Reach_j, num_days):
    """
    Calculate the Total Phosphorus (TP) load in tons for a given reach over a specified number of days.

    Parameters:
    Reach_j: A Pandas's Dataframe containing the following columns:
        - 'SedPOut': Sediment phosphorus output (units unspecified).
        - 'SolPOut': Soluble phosphorus output (units unspecified).
        - 'FlowOut': Flow output (units unspecified).
        - 'FlowOutFraction': Fraction of the flow output (units unspecified).
        - 'SedPIn': Sediment phosphorus input (units unspecified).
        - 'SolPIn': Soluble phosphorus input (units unspecified).
        - 'FlowIn': Flow input (units unspecified).
        - 'FlowInFraction': Fraction of the flow input (units unspecified).
    num_days (int): The number of days over which to calculate the TP load.

    Returns:
    numpy.ndarray: An array of TP load in tons per day for the specified number of days.
    """
    #TP Daily loadings (tons/day)
    TP_Load_tons = np.zeros(num_days)
    TP_Load_tons = ((Reach_j['SedPOut'] + Reach_j['SolPOut']) \
                    *(Reach_j['FlowOut']*Reach_j['FlowOutFraction'])*(3600 *24 * 1000/1E9)) \
                    -((Reach_j['SedPIn'] + Reach_j['SolPIn']) \
                    *(Reach_j['FlowIn']*Reach_j['FlowInFraction'])*(3600 * 24 * 1000/1E9))
    return TP_Load_tons


# ## calculate_TN_Load_tons
def calculate_TN_Load_tons(Reach_j, num_days):
    """
    Calculate the total nitrogen (TN) load in tons for a given reach over a specified number of days.

    Parameters:
    Reach_j: A Pandas's Dataframe containing the following columns:
        - 'SolNO3Out': Soluble nitrate outflow concentration.
        - 'SolNH4Out': Soluble ammonium outflow concentration.
        - 'SolOrgNOut': Soluble organic nitrogen outflow concentration.
        - 'SedNH4Out': Sediment-bound ammonium outflow concentration.
        - 'SedOrgNOut': Sediment-bound organic nitrogen outflow concentration.
        - 'FlowOut': Outflow volume.
        - 'FlowOutFraction': Fraction of the outflow.
        - 'SolNO3In': Soluble nitrate inflow concentration.
        - 'SolNH4In': Soluble ammonium inflow concentration.
        - 'SolOrgNIn': Soluble organic nitrogen inflow concentration.
        - 'SedNH4In': Sediment-bound ammonium inflow concentration.
        - 'SedOrgNIn': Sediment-bound organic nitrogen inflow concentration.
        - 'FlowIn': Inflow volume.
        - 'FlowInFraction': Fraction of the inflow.
    num_days (int): The number of days over which to calculate the TN load.

    Returns:
    np.ndarray: An array of TN load values in tons for each day.
    """
    #TN Daily loadings (tons/day)
    TN_Load_tons = np.zeros(num_days)
    TN_Load_tons = ((Reach_j['SolNO3Out'] + Reach_j['SolNH4Out'] + Reach_j['SolOrgNOut'] + Reach_j['SedNH4Out'] + Reach_j['SedOrgNOut'])\
                                          *(Reach_j['FlowOut']*Reach_j['FlowOutFraction'])*(3600 * 24 *1000/1E9))\
                                          -((Reach_j['SolNO3In'] + Reach_j['SolNH4In'] + Reach_j['SolOrgNIn'] + Reach_j['SedNH4In'] + Reach_j['SedOrgNIn'])\
                                            *(Reach_j['FlowIn']*Reach_j['FlowInFraction'])*(3600 * 24 * 1000/1E9))
    return TN_Load_tons


# 
def computeReach_TP_TN_Loads(input_data_files, All_reaches, 
                             Reach_k_daily_TP_df, 
                             Reach_k_daily_TN_df, 
                             Reach_k_daily_TP_TN_df, 
                             num_days,
                             myNaNCols): 
                                
    """
    Populates the daily flow data for each reach and updates the lists of Reach IDs and downstream Reach IDs.
    Args:
        input_data_files (str): The directory path where the input data files are located.
        All_reaches (list): A list of filenames for each reach's data.
        Reach_j_daily_df (pd.DataFrame): A DataFrame to store the daily flow data for each reach.
        Reach_IDs (list): A list to store the Reach IDs.
        DS_Reach (list): A list to store the downstream Reach IDs.
        num_days (int): The number of days for which the flow data is to be calculated.
    Returns:
        pd.DataFrame: Updated DataFrame with daily flow data for each reach.
        list: Updated list of Reach IDs.
        list: Updated list of downstream Reach IDs.
    """
    num_reaches = len(All_reaches)
    for j in range(num_reaches):
        Reach_j = pd.read_csv(os.path.join(input_data_files, All_reaches[j]))
        Reach_j = Reach_j.iloc[1:, :]
        
        for col in myNaNCols:
            Reach_j = handleNaNValues(Reach_j, col)

        # Collect Flow data for each reach
        TP_Load_tons = calculate_TP_Load_tons(Reach_j, num_days)
        TN_Load_tons = calculate_TN_Load_tons(Reach_j, num_days)

        # update Reach_k_daily_TP_df with TP_Load_tons
        
        Reach_k_daily_TP_df['%s_%s'%(Reach_j['ReachID'].iloc[0], 
                                Reach_j['ReachNextID'].iloc[0])] = TP_Load_tons
        
        # update Reach_k_daily_TN_df with TN_Load_tons
        Reach_k_daily_TN_df['%s_%s'%(Reach_j['ReachID'].iloc[0], 
                                Reach_j['ReachNextID'].iloc[0])] = TN_Load_tons
        
        if 'Daily_TP_Load(tons)_%s'%Reach_j['ReachID'].iloc[0] in Reach_k_daily_TP_TN_df.columns: #check if the column exists 
            Reach_k_daily_TP_TN_df['Daily_TP_Load(tons)_%s'%Reach_j['ReachID'].iloc[0]] = Reach_k_daily_TP_TN_df['Daily_TP_Load(tons)_%s'%Reach_j['ReachID'].iloc[0]] + TP_Load_tons
            Reach_k_daily_TP_TN_df['Daily_TN_Load(tons)_%s'%Reach_j['ReachID'].iloc[0]] = Reach_k_daily_TP_TN_df['Daily_TN_Load(tons)_%s'%Reach_j['ReachID'].iloc[0]] + TN_Load_tons

        else:
            Reach_k_daily_TP_TN_df['Daily_TP_Load(tons)_%s'%Reach_j['ReachID'].iloc[0]] = TP_Load_tons
            Reach_k_daily_TP_TN_df['Daily_TN_Load(tons)_%s'%Reach_j['ReachID'].iloc[0]] = TN_Load_tons
    
    return Reach_k_daily_TP_df, Reach_k_daily_TN_df, Reach_k_daily_TP_TN_df
# ## check_string_in_columns


def check_string_in_columns(df, string):
    """
    Check if a string exists in any of the column names of a DataFrame.

    Parameters:
    df (DataFrame): Pandas DataFrame to search within.
    string (str): String to search for.

    Returns:
    found (bool): True if the string exists in any column name, False otherwise.
    """

    found = any(column.startswith(string) for column in df.columns)
    return found

def check_string_in_rows(df, string):
    """
    Check if a string exists in any of the row values of a DataFrame.

    Parameters:
    df (DataFrame): Pandas DataFrame to search within.
    string (str): String to search for.

    Returns:
    found (bool): True if the string exists in any row value, False otherwise.
    """
    found = any(string in row for row in df)
    return found


# ## downsamplingDF


def downsamplingDF(Reach_k_daily_df):
    """
    Source: https://www.datacamp.com/tutorial/pandas-resample-asfreq
    
    Downsample a daily DataFrame to monthly and annual aggregations, 
    and processes the annual DataFrame.
    Parameters:
    Reach_k_daily_df (pd.DataFrame): DataFrame with daily data, 
    must have a 'date' column.
    Returns:
    pd.DataFrame: Transposed annual DataFrame with processed column names 
    and 'Year' as the index.
    """
    Reach_k_daily_df = Reach_k_daily_df.set_index(['date'])

    Reach_k_daily_df.index = pd.to_datetime(Reach_k_daily_df.index, 
                                        unit = 's') # convert index to datetime

    Reach_k_monthly_df = Reach_k_daily_df.resample('M').sum()

    Reach_k_annual_df = Reach_k_monthly_df.resample('Y').sum() # aggregation


    if check_string_in_columns(Reach_k_annual_df, 'Daily_TP_Load(tons)_') : 
        Reach_k_annual_df.columns = Reach_k_annual_df.columns.str.strip('Daily__Load(tons)_') 

    if check_string_in_columns(Reach_k_annual_df, 'Daily_TN_Load(tons)_') : 
        Reach_k_annual_df.columns = Reach_k_annual_df.columns.str.strip('Daily__Load(tons)_')

    n_yrs = len(Reach_k_annual_df.index)
    # print(n_yrs)
    Year = np.zeros(n_yrs)
    for i in range(n_yrs):
        Year[i] = Reach_k_annual_df.index[i].year #avoid the for loop.
    Year = Year.astype(int)

    Reach_k_annual_df['Year'] = Year

    Reach_k_annual_df = Reach_k_annual_df.set_index('Year')

    Reach_k_annual_df = Reach_k_annual_df.T #transpose the dataframe

    Reach_k_annual_df = Reach_k_annual_df.rename(columns={0:"Reach"}) #rename the column
    # print(temp_df.info())

    return Reach_k_annual_df

def process_new_tp_tn_df(TP_df, IP_Reaches_In_Out, Fst_Yr, Lst_Yr):
    """
    Process the TP_transp_df DataFrame and merge it with IP_Reaches_In_Out DataFrame.

    Args:
        TP_transp_df (pd.DataFrame): The DataFrame to process.
        IP_Reaches_In_Out (pd.DataFrame): The DataFrame to merge with.
        Fst_Yr (int): The first year in the range.
        Lst_Yr (int): The last year in the range.

    Returns:
        pd.DataFrame: The final merged DataFrame.
    """
    # Transpose the DataFrame
    TP_transp_df = TP_df.transpose()

    # Set the axis labels
    TP_transp_df = TP_transp_df.set_axis([i for i in range(Fst_Yr, Lst_Yr + 1)], axis=1)

    # Reset the index
    TP_transp_df.reset_index(drop=False, inplace=True)

    # Set the index to 'REACH' if necessary
    if 'REACH' not in TP_transp_df.columns:
        if 'index' in TP_transp_df.columns:
            TP_transp_df.rename(columns={'index': 'REACH'}, inplace=True)

    # Format the 'REACH' column to int64 datatype
    TP_transp_df['REACH'] = TP_transp_df['REACH'].astype('int64')
    IP_Reaches_In_Out['REACH'] = IP_Reaches_In_Out['REACH'].astype('int64')

    # Merge the DataFrames
    Final_Network_df = pd.merge(TP_transp_df, 
                                IP_Reaches_In_Out, how='outer', on='REACH')

    return Final_Network_df

def create_splitting_tp_tn_df(IP_Reaches_In_Out):
    """
    Create a DataFrame for splitting TP ratios based on the IP_Reaches_In_Out DataFrame.

    Args:
        IP_Reaches_In_Out (pd.DataFrame): The DataFrame containing reach information.

    Returns:
        pd.DataFrame: The resulting DataFrame with REACH and Ratio columns.
    """
    # Create a new DataFrame with REACH column
    Splitting_TP_df = pd.DataFrame(IP_Reaches_In_Out['REACH'], columns=['REACH'])
    
    # Initialize the Ratio column with zeros
    Splitting_TP_df['Ratio'] = np.zeros(len(IP_Reaches_In_Out['REACH']))
    
    # Set the REACH column as the index
    Splitting_TP_df = Splitting_TP_df.set_index('REACH')
    
    return Splitting_TP_df

def merge_ratio_TP_TN_percentage_data(Final_Network_TP_df, 
                                      hru_df_single_obj_opti_TP, 
                                      final_columns_format):
    """
    Process the optimization data by merging DataFrames and filtering columns.

    Args:
        Final_Network_TP_df (pd.DataFrame): The DataFrame containing the final network TP data.
        hru_df_single_obj_opti_TP (pd.DataFrame): The DataFrame containing the single objective optimization TP data.
        Years (list): The list of years to be included in the final DataFrame.

    Returns:
        pd.DataFrame: The final merged and filtered DataFrame.
    """
    # Ensure 'REACH' column is of type int64
    hru_df_single_obj_opti_TP['REACH'] = hru_df_single_obj_opti_TP['REACH'].astype('int64')

    # Merge the DataFrames
    merged_df_single_obj_optim_TP = pd.merge(Final_Network_TP_df, 
                                             hru_df_single_obj_opti_TP, 
                                             on='REACH', 
                                             how='outer')

    # Remove rows where the 'REACH' column is 0
    merged_df_single_obj_optim_TP = merged_df_single_obj_optim_TP[merged_df_single_obj_optim_TP['REACH'] != 0]

    # Filter the DataFrame to keep only the specified columns
    merged_df_single_obj_optim_TP = merged_df_single_obj_optim_TP[final_columns_format]

    return merged_df_single_obj_optim_TP

def process_single_multi_obj_data(pivotT, TP_or_TN, type, columns_to_drop, columns_to_keep, output_file):
    """
    Processes single or multi-objective data from a pivot table and saves the result to a CSV file.
    Args:
        pivotT (pd.DataFrame): The pivot table containing the data to be processed.
        type (str): The type of data processing to perform. Should be either 'single_obj' or 'multi_obj'
        columns_to_drop (list): List of column names to be dropped from the result DataFrame.
        columns_to_keep (list): List of column names to be kept in the result DataFrame.
        output_file (str): The file path where the resulting CSV file will be saved.
    Returns:
        None
    """
    
    if TP_or_TN == 'TP':
        max_index = pivotT.groupby(level='REACH')['percent_TP_tons_by_REACH'].idxmax()
    else:
        max_index = pivotT.groupby(level='REACH')['percent_TN_tons_by_REACH'].idxmax()
    if type == 'multi_obj':  
       max_index = pivotT.groupby(level='REACH')['sum_percent_TP_TN'].idxmax()
    

    # Create a new DataFrame with the rows containing the maximum values for each 'Reach' group
    result_df = pivotT.loc[max_index]

    result_df.reset_index(inplace=True)
    
    # rename 'Area_ac' to 'Area_acres'
    result_df.rename(columns={'Area_ac': 'Area_acres'}, inplace=True)
    
    # print(result_df.info())
    # Drop unnecessary columns and keep only the specified columns
    result_df = result_df.drop(columns=columns_to_drop)[columns_to_keep]

    # Save the result to a CSV file
    # result_df.to_csv(output_file, index=False, header=True)

    return result_df

def update_tp_tn_df(IP_Reaches_In_Out, IP_TP_Splitting, Fst_Yr, Lst_Yr):
    """
    Update the TP_df DataFrame based on the IP_Reaches_In_Out and IP_TP_Splitting DataFrames.

    Args:
        IP_Reaches_In_Out (pd.DataFrame): DataFrame containing reach information.
        IP_TP_Splitting (pd.DataFrame): DataFrame containing TP splitting information.
        Fst_Yr (int): The first year in the range of years.
        Lst_Yr (int): The last year in the range of years.

    Returns:
        pd.DataFrame: The updated TP_df DataFrame.
    """
    # Initialize an empty DataFrame for TP_df
    TP_df = pd.DataFrame()

    # Define the range of years
    Years = range(Fst_Yr, Lst_Yr + 1)

    # Traverse the IP_Reaches_In_Out DataFrame
    for i in range(len(IP_Reaches_In_Out)):
        TP_Out = np.zeros(len(Years))
        TP_In = np.zeros(len(Years))

        # Traverse the IP_TP_Splitting DataFrame
        for reach_reach, cols in IP_TP_Splitting.iterrows():
            # If i is the first element of the Reach_Reach column in IP_TP_Splitting
            if str(IP_Reaches_In_Out['REACH'].iloc[i]) == reach_reach.split("_")[0]:
                for y in Years:
                    TP_Out[y - Fst_Yr] = TP_Out[y - Fst_Yr] + cols[y]

            # If i is the last element of the Reach_Reach column in IP_TP_Splitting
            if str(IP_Reaches_In_Out['REACH'].iloc[i]) == reach_reach.split("_")[1]:
                for y in Years:
                    TP_In[y - Fst_Yr] = TP_In[y - Fst_Yr] + cols[y]

        # Update TP_df with the difference between TP_Out and TP_In
        TP_df['%s' % IP_Reaches_In_Out['REACH'].iloc[i]] = TP_Out - TP_In

    return TP_df

class myGraph:
    """
    A class to represent a directed graph using an adjacency list.
    Attributes
    ----------
    vertices : dict
        A dictionary where keys are vertex identifiers and values are lists of adjacent vertices.
    Methods
    -------
    __init__():
        Initializes the graph with an empty dictionary of vertices.
    add_vertex(vertex):
        Adds a vertex to the graph.
    add_edge(source, target):
        Adds a directed edge from the source vertex to the target vertex.
    incoming_nodes(vertex):
        Returns a list of vertices that have edges directed towards the given vertex.
    outgoing_nodes(vertex):
        Returns a list of vertices that the given vertex has edges directed towards.
    __str__():
        Returns a string representation of the graph.
    """
    def __init__(self):
        self.vertices = {} 
    
    def add_vertex(self, vertex):
    # Set the data for the vertex
        self.vertices[vertex] = [] 
    
    def add_edge(self, source, target):
        self.vertices[source].append(target) #graph is represented as an adjacency list
    
    def incoming_nodes(self, vertex):
        incoming = []
        for v, edges in self.vertices.items():
            # it checks if the given vertex is a target node
            if vertex in edges:
                incoming.append(v) # v is the source node, 'v, vertex' is the incoming edge
        return incoming
    
    def outgoing_nodes(self, vertex):
        if vertex not in self.vertices:
            return []  # If the vertex doesn't exist in the graph, return an empty list
        return self.vertices[vertex]
    
    def __str__(self):
        result = ""
        for vertex, edges in self.vertices.items():
            result += f"{vertex}: {', '.join(edges)}\n"
        return result
    
# Function to collect data for each node and write it to the CSV file
def write_node_data_to_csv(graph, json_data, csv_filename):
    """
    Writes node data from a graph and JSON data to a CSV file.
    Args:
        graph (Graph): The graph object containing node relationships.
        json_data (dict): The JSON data containing node information.
        csv_filename (str): The filename for the output CSV file.
    The CSV file will contain the following columns:
        - SUB: The node identifier.
        - Ingoing: A comma-separated list of nodes that have edges pointing to this node.
        - Outgoing: A comma-separated list of nodes that this node points to.
    Nodes with an identifier of '0' are skipped. If a node has an outgoing edge to '0', 
    the 'Outgoing' field for that node will be empty.
    Example:
        write_node_data_to_csv(graph, json_data, 'output.csv')
    """
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['SUB', 'Ingoing', 'Outgoing']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for node in json_data['nodes']:
            SUB = node['id']
            if SUB == '0':
                continue
            ingoing = ', '.join(graph.incoming_nodes(SUB)) 
            outgoing = ', '.join(graph.outgoing_nodes(SUB))
            if '0' in graph.outgoing_nodes(SUB):
                outgoing = ''
            
            print(f"SUB: {SUB}, Ingoing: {ingoing}, Outgoing: {outgoing}")
            writer.writerow({'SUB': SUB, 'Ingoing': ingoing, 'Outgoing': outgoing})
