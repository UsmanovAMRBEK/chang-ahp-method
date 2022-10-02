import numpy as np
import pandas as pd

persons_data = np.load('./person_prep_data_1.npy')

for a in range(100):

    data = persons_data[a][0]
    for i in range(1,4):
        data+=persons_data[a][i]

    pandas_alldc = pd.DataFrame(data)
    pandas_alldc.index = ['A','B','C','D']
    pandas_alldc.columns = ['L','M','U']

    pandas_alldc_data = pandas_alldc.copy()


    def activate(pandas_alldc):
        pandas_alldc['L'] = pandas_alldc['L']*(1/pandas_alldc['L'].sum())
        pandas_alldc['M'] = pandas_alldc['M']*(1/pandas_alldc['M'].sum())
        pandas_alldc['U'] = pandas_alldc['U']*(1/pandas_alldc['U'].sum())
        return pandas_alldc

    pandas_alldc_acitve = activate(pandas_alldc)


    def row2vector(active_data):
        import numpy as np
        if active_data.size==9:
            s1 = np.array(active_data.iloc[[0]])
            s2 = np.array(active_data.iloc[[1]])
            s3 = np.array(active_data.iloc[[2]])
            return [s1,s2,s3]
        else:
            s1 = np.array(active_data.iloc[[0]])
            s2 = np.array(active_data.iloc[[1]])
            s3 = np.array(active_data.iloc[[2]])
            s4 = np.array(active_data.iloc[[3]])
            return [s1,s2,s3,s4]


    def vM2_M1(M2, M1):
        """
        Args:
            M2(l2,m2,u2)
            M1(l1,m1,u1)
        return: V(M2>=M1)
        """
        M1=M1[0]; M2=M2[0]
        if M2[1]>=M1[1]:
            return 1
        elif M1[0]>M2[2]:
            return 0
        else:
            return (M1[0]-M2[2])/( (M2[1]-M2[2]) - (M1[1]-M1[0]) )



    def final_data(active_data):
        if active_data.size==9:
            s1, s2 ,s3 = row2vector(active_data)

            s1_s2 = vM2_M1(s1,s2)
            s1_s3 = vM2_M1(s1,s3)
            min_s1 = min(s1_s2,s1_s3)

            s2_s1 = vM2_M1(s2,s1)
            s2_s3 = vM2_M1(s2,s3)
            min_s2 = min(s2_s1, s2_s3)

            s3_s1 = vM2_M1(s3,s1)
            s3_s2 = vM2_M1(s3,s2)
            min_s3 = min(s3_s1,s3_s2)

            final_data_s1 = pd.DataFrame([s1_s2,s1_s3,min_s1], index=['s1>s2', 's1>s3','min_s1'])
            final_data_s2 = pd.DataFrame([s2_s1,s2_s3,min_s2], index=['s2>s1', 's2>s3','min_s2'])
            final_data_s3 = pd.DataFrame([s3_s1,s3_s2,min_s3], index=['s3>s1', 's3>s2','min_s3'])
            return final_data_s1,final_data_s2,final_data_s3
        else:
            s1, s2 ,s3, s4 = row2vector(active_data)

            s1_s2 = vM2_M1(s1,s2)
            s1_s3 = vM2_M1(s1,s3)
            s1_s4 = vM2_M1(s1,s4)
            min_s1 = min(s1_s2,s1_s3,s1_s4)


            s2_s1 = vM2_M1(s2,s1)
            s2_s3 = vM2_M1(s2,s3)
            s2_s4 = vM2_M1(s2,s4)
            min_s2 = min(s2_s1,s2_s3,s2_s4)


            s3_s1 = vM2_M1(s3,s1)
            s3_s2 = vM2_M1(s3,s2)
            s3_s4 = vM2_M1(s3,s4)
            min_s3 = min(s3_s1,s3_s2,s3_s4)


            s4_s1 = vM2_M1(s4,s1)
            s4_s2 = vM2_M1(s4,s2)
            s4_s3 = vM2_M1(s4,s3)
            min_s4 = min(s4_s1,s4_s2,s4_s3)

            final_data_s1 = pd.DataFrame([s1_s2,s1_s3,s1_s4,min_s1], index=['s1>s2', 's1>s3','s1>s4', 'min_s1'])
            final_data_s2 = pd.DataFrame([s2_s1,s2_s3,s2_s4,min_s2], index=['s2>s1', 's2>s3','s2>s4','min_s2'])
            final_data_s3 = pd.DataFrame([s3_s1,s3_s2,s3_s4,min_s3], index=['s3>s1', 's3>s2','s3>s4','min_s3'])
            final_data_s4 = pd.DataFrame([s4_s1,s4_s2,s4_s3,min_s4], index=['s4>s1', 's4>s2','s4>s3','min_s4'])
            return final_data_s1,final_data_s2,final_data_s3,final_data_s4



    f1, f2, f3, f4 = final_data(pandas_alldc_acitve)


    sum_min = f1.iloc[[2]].values+f2.iloc[[2]].values+f3.iloc[[2]].values+f4.iloc[[2]].values
    sum_min_pd = pd.DataFrame(sum_min,index=['sum_min'])

    lst = [[int(f1.iloc[[2]].values/sum_min),int(f2.iloc[[2]].values/sum_min),int(f3.iloc[[2]].values/sum_min),int(f4.iloc[[2]].values/sum_min)]]
    d_result = pd.DataFrame(lst,index=list('d'))

    def multiple_dfs(df_list, sheets, file_name, spaces):
        writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
        row = 0
        for dataframe in df_list:
            dataframe.to_excel(writer,sheet_name=sheets,startrow=row , startcol=0)   
            row = row + len(dataframe.index) + spaces + 1
        writer.save()

    # list of dataframes
    dfs = [pandas_alldc_data,pandas_alldc_acitve,f1,f2,f3,f4, sum_min_pd, d_result]


    def test(d_result):
        for i in d_result:
            if i % 1 != 0:
                return True
        return False


    # run function
    if test(d_result):
        multiple_dfs(dfs, 'Validation', f'test_one_two.xlsx', 1)