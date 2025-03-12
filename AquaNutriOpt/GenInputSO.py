import time
import pandas as pd
import os
import sys

def gen_input_so(time_period: int):
    """Generates the input files for single-objective optimization based on the time period specified.
    
    Args:
        time_period (int): The time period for which to generate the input files.
    """

    def BMP_Prep(TimePeriod):
        file = open('BMPsInfo.csv')
        file.readline()
        Table = []
        while True:
            l = file.readline()
            if l == '':
                break
            l = l.strip('\n')
            l = l.split(',')
            Table.append([int(l[0]), l[2], float(l[3]), float(l[4]),
                        float(l[5])])  # float(l[3] is N reduction cap, float(l[4]) is P reduction cap.
        file.close()

        def FindBMPs(Source, LU_code):
            L = []
            for i in range(len(Source)):
                if Table[i][0] == LU_code:
                    L.append(i)
            return L
            # ------------------------------------------------------------

        file = open('NetworkInfo.csv')
        filep = open('Net_Data_old.csv', 'w+')
        BMP = open('BMP_Tech.csv', 'w+')
        BMP.write('BMPs,Cost,P_LB,N_LB,P_UB,N_UB\n')

        file.readline()
        while True:
            l = file.readline()

            if l == '':
                break
            lp = l.strip('\n')
            lp = lp.split(',')

            ''' In code below, lp[ ] and lp[ ] indexing is where the code must be changed in order to incorporate the time index.'''

            LU_codes = [i for i in lp[4 + 2 * TimePeriod].split(' ') if (i.strip('.0')).isdigit()]
            #print(LU_codes)
            LU_Areas = [i for i in lp[5 + 2 * TimePeriod].split(' ') if (i.strip('.0')).isdigit()]
            # LU_percent = [i for i in lp[8].split(' ') if (i.strip('.0').isdigit())]  #####
            ####Table1.append[int(lp[0], int(lp[6], int(lp[8])))]
            if len(LU_codes) > 0:
                l = l.strip('\n')
                l = l + ','
                for i in range(len(LU_codes)):

                    List = FindBMPs(Table, int(float(LU_codes[i])))
                    for j in List:
                        l = l + ' ' + Table[j][1] + '_' + lp[0]

                        # print('Table j1', Table[j][1])
                        # print('lp0', lp[0])
                        # print('Table j3', Table[j][3])
                        # print('lp percent area', lp[6 + 2 * TimePeriod])
                        # print('Table j2', Table[j][2])
                        # print('Table j4', Table[j][4])
                        # print('LU area', LU_Areas[i])
                        #print("*******************************")
                        # BMP.write(Table[j][1] + '_' + lp[0] + ',' + str(Table[j][4] * float(LU_Areas[i])) + ',' + str(
                        #     Table[j][3] * float(lp[6 + 2 * TimePeriod])) + ',' + str(
                        #     Table[j][2] * float(lp[6 + 2 * TimePeriod + 1])) + ',' + str(
                        #     Table[j][3] * float(lp[6 + 2 * TimePeriod])) + ',' + str(
                        #     Table[j][2] * float(lp[6 + 2 * TimePeriod + 1])) + '\n')
                        BMP.write(
                            Table[j][1] + '_' + lp[0] + ',' + str(
                                Table[j][4] * float(LU_Areas[i]))+ ',' + str(Table[j][3] * float(lp[6 + 2 * TimePeriod])) + ',' + str(
                                Table[j][2] * float(lp[6 + 2 * TimePeriod])) + ',' + str(Table[j][3] * float(lp[6 + 2 * TimePeriod])) + ',' + str(
                                Table[j][2] * float(lp[6 + 2 * TimePeriod]))+'\n')
                        ## Table[j][3] is phosphorous redn cap, Table[j][2] is nitrogen redn cap, Table[j][4] is cost
                        ## Both P and N red cap is multiplied by percent area of dominant land use (within total Reach Area).
                l = l + '\n'

            filep.write(l)

        BMP.close()
        filep.close()
        file.close()

    BMP_Prep(time_period)


    def SplittingNodeCreation(TimePeriod):
        # Specify the file path for the CSV file
        file_path = 'Net_Data_old.csv'

        # List of column indices to be deleted (0-based index) #different for single and multiobjective
        columns_to_delete = [-4,-3,-2]
        # For example, the script above deletes the 2nd last, 3rd last, and 4th last columns for multiobjective optimization case

        # Read the CSV file into a DataFrame without header
        df = pd.read_csv(file_path, header=None,dtype=object)

        # Drop the specified columns using the column indices
        df.drop(df.columns[columns_to_delete], axis=1, inplace=True)

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False, header=False)

        #time.sleep(1)
        while not os.path.exists('Net_Data_old.csv'):
            time.sleep(1)

        file = open('Net_Data_old.csv')
        filep = open('Net_Data_split.csv', 'w+')
        splitNodeIncomings = {}
        while True:
            l = file.readline()

            if l == '':
                break
            lp = l.strip('\n')
            lp = lp.split(',')
            splitRatio = lp[3]
            splitRatio = splitRatio.split(' ')
            ccc = False
            if len(splitRatio) <= 1:
                filep.write(l)
            else:
                splitNodeIncomings[lp[0]] = lp[1]
                for iii in range(2 * TimePeriod):
                    if float(lp[4 + iii]) > 0 or float(lp[4 + iii]) < 0:
                        ccc = True
                        if ccc == True:
                            break
                if ccc == True:
                    filep.write(lp[0] + '_s,' + lp[1] + ',' + lp[0] + ',,')
                    #print("nodes is", lp[0])
                    for iii in range(2 * TimePeriod):
                        if float(lp[4 + iii]) > 0 or float(lp[4 + iii]) < 0:
                            filep.write(lp[4 + iii] + ',')
                        else:
                            filep.write('0,')
                    filep.write(lp[4 + 2 * TimePeriod] + '\n')
                    filep.write(lp[0] + ',' + lp[0] + '_s' + ',' + lp[2] + ',' + lp[3] + ',')
                    for iii in range(2 * TimePeriod):
                        filep.write('0,')
                    filep.write(lp[4 + 2 * TimePeriod] + '\n')
                else:
                    filep.write(l)
        filep.close()
        file.close()
        keys = []
        values = []
        ingoings_list = []
        for key, value in splitNodeIncomings.items():
            keys.append(key)
            values.append(value.split(' '))
        for i in range(len(values)):
            ingoings_list += values[i]
        #print(ingoings_list)

        #time.sleep(1)
        while not os.path.exists('Net_Data_split.csv'):
            time.sleep(1)
            pass

        file = open('Net_Data_split.csv')
        filep = open('Net_Data_split2.csv', 'w+')
        while True:
            l2 = file.readline()
            if l2 == '':
                break
            lp2 = l2.strip('\n')
            lp2 = lp2.split(',')

            if lp2[0] not in ingoings_list:
                # print('True')
                filep.write(l2)
            else:
                if lp2[0] not in keys:  ##because the ingoings lp[2] might have multiple outgoings as well. write code for else condition
                    # print("True")
                    filep.write(lp2[0] + ',' + lp2[1] + ',' + lp2[2] + '_s,' + lp2[3] + ',')
                    for iii in range(2 * TimePeriod):
                        filep.write(lp2[4 + iii] + ',')
                    filep.write(lp2[4 + 2 * TimePeriod] + '\n')
                else:
                    abc = []
                    temp_outgoing = lp2[2].split(' ')
                    print(temp_outgoing)
                    for items in temp_outgoing:
                        if items in keys:
                            abc.append(items + '_s')
                        else:
                            abc.append(items)
                    outgoing = " ".join(abc)
                    filep.write(lp2[0] + ',' + lp2[1] + ',' + outgoing + ',' + lp2[3] + ',')   #lp2[2] + '_s,'
                    for iii in range(2 * TimePeriod):
                        filep.write('0,')
                    filep.write(lp2[4 + 2 * TimePeriod] + '\n')

        file.close()
        filep.close()

    SplittingNodeCreation(time_period)

    while not os.path.exists('Net_Data_split2.csv'):
        pass

    def Prep_Net(TimePeriod):
        file = open('Net_Data_split2.csv')
        filep = open('Net_Data.csv', 'w+')
        """ writing the header of the Net_Data file"""
        filep.write('Reach,Ingoings,Outgoings,Split Ratio,')
        for iii in range(TimePeriod):
            filep.write('P_{}'.format(iii) + ',')
        for iii in range(TimePeriod):
            filep.write('N_{}'.format(iii) + ',')
        filep.write('BMPs\n')

        l = file.readline()
        filep.write(l)

        counter = 0
        while True:
            l = file.readline()
            if l == '':
                break

            lp = l.split(',')
            """lp[6] for single timeperiod changes to lp[4+2*Timeperiod]"""
            BMPs = lp[4 + 2 * TimePeriod].strip('\n')
            BMPs = BMPs.split(' ')
            for j in range(len(BMPs)):
                BMPs[j] = BMPs[j].strip(' ')

            ii = 0
            while ii < len(BMPs):
                if BMPs[ii] == '':
                    del (BMPs[ii])
                else:
                    ii += 1

            InNodes = lp[1].strip('\n')
            InNodes = InNodes.split(' ')
            for j in range(len(InNodes)):
                InNodes[j] = InNodes[j].strip(' ')

            ii = 0
            while ii < len(InNodes):
                if InNodes[ii] == '':
                    del (InNodes[ii])
                else:
                    ii += 1

            """ Making an separate array of Phosphorous and Nitrogen loading each for user input TimePeriods. """
            phosphorousTimed = []
            nitrogenTimed = []
            for iii in range(TimePeriod):
                phosphorousTimed.append(lp[4 + iii])
                nitrogenTimed.append(lp[4 + TimePeriod + iii])

            if len(BMPs) >= 1 and len(InNodes) >= 1:
                """lp[4] means Phosphorous loading at that Node"""
                jjj = False
                for iii in range(2 * TimePeriod):
                    if float(lp[4 + iii]) < 0:
                        jjj = True
                if jjj == True:
                    filep.write(lp[0] + ',')
                    filep.write(lp[1] + ' ' + lp[0] + '_0' + ' ' + lp[0] + '_a,')
                    filep.write(lp[2] + ',')
                    filep.write(lp[3] + ',')
                    for iii in range(2 * TimePeriod):
                        # if float(lp[4+iii]) <= 0:
                        filep.write('0,')
                    filep.write('\n')
                    # else:
                    # filep.write(lp[4+iii]+',')
                    # for iii in range(TimePeriod):
                    # if float(lp[4+TimePeriod+iii]) <= 0:
                    # filep.write('0,')
                    # else:
                    # filep.write(lp[4+TimePeriod+iii]+',')

                    filep.write(lp[0] + '_0,,' + lp[0] + ',,')
                    for iii in range(2 * TimePeriod):
                        if float(lp[4 + iii]) <= 0:
                            filep.write('0,')
                        else:
                            filep.write(lp[4 + iii] + ',')
                    filep.write(lp[4 + 2 * TimePeriod])  #### removed \n from here.

                    filep.write(lp[0] + '_a,,' + lp[0] + ',,')
                    # print("lp[0]+_a is", lp[0]+'_a')

                    for iii in range(2 * TimePeriod):

                        if float(lp[4 + iii]) <= 0:
                            filep.write(lp[4 + iii] + ',')
                        else:
                            filep.write('0,')
                    filep.write('\n')

                else:
                    filep.write(lp[0] + ',')
                    filep.write(lp[1] + ' ' + lp[0] + '_0,')
                    filep.write(lp[2] + ',')
                    filep.write(lp[3] + ',')
                    for iii in range(2 * TimePeriod):
                        filep.write('0,')
                    filep.write('\n')

                    filep.write(lp[0] + '_0,,' + lp[0] + ',,')
                    for iii in range(2 * TimePeriod):
                        filep.write(lp[4 + iii] + ',')
                    filep.write(lp[4 + 2 * TimePeriod])  ##removed \n from here previously : filep.write(lp[4 + 2 * TimePeriod] + '\n')
            else:
                jjj = False
                #print("node is:", lp[0] )
                for iii in range(2 * TimePeriod):
                    if float(lp[4 + iii]) < 0:
                        jjj = True
                if jjj == True:
                    filep.write(lp[0] + ',')
                    if lp[1] == '':
                        filep.write(lp[0] + '_a,')
                    else:
                        filep.write(lp[0] + '_a' + ' ' + lp[1] + ',')
                    filep.write(lp[2] + ',')
                    filep.write(lp[3] + ',')
                    for iii in range(2 * TimePeriod):
                        if float(lp[4 + iii]) <= 0:
                            filep.write('0,')
                        else:
                            filep.write(lp[4 + iii] + ',')
                    filep.write('\n')
                    filep.write(lp[0] + '_a,,' + lp[0] + ',,')
                    for iii in range(2 * TimePeriod):
                        if float(lp[4 + iii]) <= 0:
                            filep.write(lp[4 + iii] + ',')
                        else:
                            filep.write('0,')
                    filep.write('\n')

                else:
                    filep.write(l)

            counter += 1
        #    if counter == 5:
        #        break

        filep.close()
        file.close()

        try:
            os.remove('Net_data_old.csv')
            os.remove('Net_Data_split.csv')
            os.remove('Net_Data_split2.csv')
        except:
            print("No such file in the directory")

    Prep_Net(time_period)


if __name__ == "__main__":
    time_period = int(sys.argv[1])
    gen_input_so(time_period)
