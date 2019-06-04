#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:23:43 2019

@author: abhi
"""

import statistics as stat
import pandas as pd
import numpy as np
import collections
from functools import reduce
import math

def drop_row_preproc(df,col_name):

    tmp0 = df[col_name] == '?'
    idx = list(tmp0[tmp0 == True].index)
    df.drop(df.index[idx])
    return df
##########################################################################################################

def gaussian_discr(df,col_name):
    
    group = df.groupby('Output')[col_name].unique()
    b = group[group.apply(lambda x: len(x)>1)]
    
    sd_ge_50K = stat.stdev(b['>50K'])
    mean_ge_50K = stat.mean(b['>50K'])
    
    sd_le_50K = stat.stdev(b['<=50K'])
    mean_le_50K = stat.mean(b['<=50K'])

    a = [mean_le_50K,sd_le_50K,mean_ge_50K,sd_ge_50K]
    return a
########################################################################################################

def preprocess_max_freq(df2,col_name):
    tmp_mode = stat.mode(df2[col_name])
    #Assumption is ? mark denotes missing values
    df[col_name].replace('?',tmp_mode, inplace=True)
    return df2

################################################################################################

def discretize_col(df2,col_name,bin_size):

    #Can be fed bin list or bin size
    df2[col_name] = pd.cut(df2[col_name],bin_size)

    return df2

################################################################################################

def train_naive_bayes(df):

    a = []

    #a is a list which is used to get all unique/distinct terms in a attribute (column).
    for each in df.columns:
        a.append(sorted(df[each].unique().tolist()))

    d = []

    #d is a dictionary which stores sub-class/sub-attribute category - (eg - private,federal,etc) - as key and initialize with 0 probability (Less than 50k).
    for each in df.columns:
        tm = sorted(df[each].unique().tolist())
        d.append(collections.OrderedDict(zip(tm,np.zeros(len(tm)).tolist())))

    #set conditional probabilities to values in dictionary with sub-class category keys.
    #For No class in output/label.
    for col_no in range (0,len(df.columns)-1):
        for idx,each in enumerate(a[col_no]):

            tmp_bool2 = ( df['Output'] == '<=50K')

            tmp_bool1 = ((df[df.columns[col_no]] == each) & ( df['Output'] == '<=50K'))

            try:
                t1 = tmp_bool1.value_counts()[1]
            except:
                t1 = 0
            ke = list(d[col_no].keys())[idx]
            d[col_no][ke] = (t1 / (tmp_bool2.value_counts()[1]))

#############################
    # Init v for storing conditional probabilities (greater than 50k).
    c = []

    for each in df.columns:
        c.append(sorted(df[each].unique().tolist()))

    v = []

    for each in df.columns:
        tm = sorted(df[each].unique().tolist())
        v.append(collections.OrderedDict(zip(tm,np.zeros(len(tm)).tolist())))

#For No class in output/label.
    for col_no in range (0,len(df.columns)-1):
        for idx,each in enumerate(c[col_no]):

            tmp_bool2 = ( df['Output'] == '>50K')

            tmp_bool1 = ((df[df.columns[col_no]] == each) & ( df['Output'] == '>50K'))

            try:
                t1 = tmp_bool1.value_counts()[1]
            except:
                t1 = 0
            ke = list(v[col_no].keys())[idx]
            v[col_no][ke] = (t1 / (tmp_bool2.value_counts()[1]))

    z3 = len(df.columns)-1

    #Setting last column probability as total probability of yes or no
    v[z3][list(v[z3].keys())[0]] = (df['Output'].value_counts()['<=50K']) / df['Output'].count()
    v[z3][list(v[z3].keys())[1]] = (df['Output'].value_counts()['>50K']) / df['Output'].count()

    #Setting last column probability as total probability of yes or no
    d[z3][list(v[z3].keys())[0]] = (df['Output'].value_counts()['<=50K']) / df['Output'].count()
    d[z3][list(v[z3].keys())[1]] = (df['Output'].value_counts()['>50K']) / df['Output'].count()

    return d, v

############################################################################################3

def test_naive_bayes(df4, p_cond_le_50k, p_cond_ge_50k, gaussian_sd_mean={}):

    col_to_be_discr = {'age', 'fnlwgt','capital-gain','capital-loss','hours-per-week'}

    #Assuming that test-data has no missing values but is not descritized
    for col_name in col_to_be_discr:
        if col_name in list(gaussian_sd_mean.keys()):
            pass
        else:
            for each in df4[col_name].unique():
                zz = list(p_cond_le_50k[(df4.columns.get_loc(col_name))].keys())
                for i in range(0,len(zz)):
                    if (each in zz[i]):
                        df4[col_name].replace(each,zz[i], inplace=True)

    out = []
    out1 = []
    out2 = []
    df_temp1 = df4.copy()
    df_temp2 = df4.copy()

    #Less than 50K Probability
    for idx,col_name in enumerate(df4.columns):
        for key in p_cond_le_50k[idx].keys():
            if col_name in list(gaussian_sd_mean.keys()):
                for i in df[col_name].unique():
                    mean = gaussian_sd_mean[col_name][0]
                    sd = gaussian_sd_mean[col_name][1]
                    ans = math.exp(-((i-mean)*(i-mean))/(2*sd*sd))/(math.sqrt(2*3.14)*sd)
                    df_temp1[col_name].replace(i,ans,inplace=True)
                    
            else:
                df_temp1[col_name].replace(key,p_cond_le_50k[idx][key],inplace=True)

    for each1 in range (0,len(df_temp1)):
        z4 = (df_temp1.iloc[each1].tolist())
        z4.append(p_cond_le_50k[14]['<=50K'])
        out1.append(reduce(lambda a, b:a*b,z4))

    #Greater than 50K Probability

    for idx,col_name in enumerate(df4.columns):
        for key in p_cond_ge_50k[idx].keys():
            if col_name in list(gaussian_sd_mean.keys()):
                for i in df[col_name].unique():
                    mean = gaussian_sd_mean[col_name][2]
                    sd = gaussian_sd_mean[col_name][3]
                    ans = math.exp(-((i-mean)*(i-mean))/(2*sd*sd))/(math.sqrt(2*3.14)*sd)
                    df_temp2[col_name].replace(i,ans,inplace=True)
                    
            else:
                df_temp2[col_name].replace(key,p_cond_ge_50k[idx][key],inplace=True)

    for each2 in range (0,len(df_temp2)):
        z5 = (df_temp2.iloc[each2].tolist())
        z5.append(p_cond_ge_50k[14]['>50K'])
        out2.append(reduce(lambda x, y:x*y,z5))

    out = np.array(out1)/np.array(out2)

    return out
########################################################################################################

def cross_valid(df5, k):

    k_outs_le_50k_p = []
    k_outs_ge_50k_p = []

    acc_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1measure_sum = 0

    for i in range (0,k):
        #Random selection with replacement for k-1/k part

        tmp1_train = df5.sample(frac=(k-1)/k,replace=True)
        tmp1_test = df5.sample(frac=(1/k),replace=True)

        col_to_be_discr = {'age':15, 'fnlwgt':67,'capital-gain':18,'capital-loss':16,'hours-per-week':17}

        for each in col_to_be_discr:
            tmp1_train = discretize_col(tmp1_train,each,col_to_be_discr[each])

        #Training model on random set
        p_tmp1_le_50k, p_tmp1_ge_50k = train_naive_bayes(tmp1_train)

        #Remove Output column from test set
        tmp2_y_actual = tmp1_test['Output']
        tmp2 = tmp1_test.drop(['Output'],1)

        #Testing with generated models on (1/k) part
        y_tmp_test = test_naive_bayes(tmp2,p_tmp1_le_50k,p_tmp1_ge_50k)

        y_tmp_pred = []

        for each in y_tmp_test:
            if each > 1:
                y_tmp_pred.append('<=50K')
            else:
                y_tmp_pred.append('>50K')

        #Test general accuracy and stuff of model
        conf_matrix,accuracy = confusion_matrix(y_tmp_pred,tmp2_y_actual.tolist())
        precision,recall = prec_recall(conf_matrix)
        F1_Measure = f1Measure(precision,recall)

        #Adding the ith iteration values to calculate sum
        acc_sum += accuracy
        precision_sum += precision
        recall_sum += recall
        f1measure_sum += F1_Measure

        print("Accuracy for iteration: "+str(i))
        print(accuracy*100)
        print("")
        print("Confusion Matrix for iteration: "+str(i))
        print(conf_matrix)
        print("")
        print("Precision & Recall for iteration: "+str(i))
        print(str(precision)+" & "+str(recall))
        print("")
        print("F1 Measure for iteration: "+str(i))
        print(F1_Measure)
        print("")

        #Save models for further use.
        k_outs_le_50k_p.append(p_tmp1_le_50k)
        k_outs_ge_50k_p.append(p_tmp1_ge_50k)

    print("Average Accuracy")
    print(acc_sum*100/k)
    print("")
    print("Average Precision & Recall")
    print(str(precision_sum/k)+" & "+str(recall_sum/k))
    print("")
    print("Average F1 Measure")
    print(f1measure_sum/k)
    print("")

    return k_outs_ge_50k_p,k_outs_le_50k_p

###########################################################################################################

def confusion_matrix(y_pred,y_actual):
    act = np.zeros(len(y_actual))
    pred = np.zeros(len(y_pred))

    for idx,each in enumerate(y_pred):
        if each == '<=50K':
            pred[idx] = 0
        else:
            pred[idx] = 1

    for idx,each in enumerate(y_actual):
        if each == '<=50K':
            act[idx] = 0
        else:
            act[idx] = 1

    #Assumption that there are only 2 output classes
    cm = np.zeros((2,2))
    for a,p in zip(act,pred):
        cm[int(a),int(p)] += 1

    acc = (act == pred).sum()/len(act)

    return cm,acc
############################################################################################################

def prec_recall(conf_matrix):

    #For true positive >50k is positive
    precision = conf_matrix[1][1]/(conf_matrix[1][1] + conf_matrix[1][0])
    recall = conf_matrix[1][1]/(conf_matrix[1][1] + conf_matrix[0][1])

    return precision,recall

############################################################################################################

def f1Measure(prec,rec):

    f1 = (prec*rec)/(prec + rec)
    return f1
############################################################################################################

def diff_test_set(df5,k_outs_ge_50k_p,k_outs_le_50k_p,gaussian_sd_mean={}):

    y_agg_pred_avg = [0 for i in range(0,len(df5))]
    y_agg_pred = []

    for idx,each in enumerate(k_outs_ge_50k_p):

        #Training model on random set
        p_tmp1_le_50k = k_outs_le_50k_p[idx]
        p_tmp1_ge_50k = k_outs_ge_50k_p[idx]

        #Remove Output column from test set
        try:
            tmp2 = df5.drop(['Output'],1)
        except:
            pass

        #Testing with generated models on (1/k) part
        y_tmp_test = test_naive_bayes(tmp2,p_tmp1_le_50k,p_tmp1_ge_50k,gaussian_sd_mean)

        y_tmp_pred = []

        for each in y_tmp_test:
            if each > 1:
                y_tmp_pred.append('<=50K')
            else:
                y_tmp_pred.append('>50K')

        #Store the pred of iteration k in a aggregate list.
        y_agg_pred.append(y_tmp_pred)

        #Setting value -1 for le50k and +1 ge50k
        for idx,each in enumerate(y_tmp_test):
            if each > 1:
                y_agg_pred_avg[idx] += -1
            else:
                y_agg_pred_avg[idx] += 1

    y_agg_pred_avg_final = []

    for each in y_agg_pred_avg:
        if each > 0:
            y_agg_pred_avg_final.append('>50K')
        else:
            y_agg_pred_avg_final.append('<=50K')
    
    return y_agg_pred_avg_final

######################################################################################################

def simple_test_acc(y_tmp_pred,tmp2_y_actual):
    
    #Test general accuracy and stuff of model
    conf_matrix,accuracy = confusion_matrix(y_tmp_pred,tmp2_y_actual.tolist())
    precision,recall = prec_recall(conf_matrix)
    F1_Measure = f1Measure(precision,recall)

    print("Accuracy")
    print(accuracy*100)
    print("")
    print("Confusion Matrix")
    print(conf_matrix)
    print("")
    print("Precision & Recall")
    print(str(precision)+" & "+str(recall))
    print("")
    print("F1 Measure")
    print(F1_Measure)
    print("")

    return 0
#######################################################################################################

df =pd.read_csv('Adult_conv_csv - adult_data.csv')

#preprocess(df,col_name)

col_to_be_pre = ['workclass','native-country','occupation']

for each in col_to_be_pre:
    df = preprocess_max_freq(df,each)

#OR Drop-Row below or Mode pre-processing above

"""
for each in col_to_be_pre:
    df = drop_row_preproc(df,each)
"""

#Gaussian code un-comment to use gaussian
gaussian_sd_mean = collections.OrderedDict()

"""
discr_list = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
for eac in discr_list:
    gaussian_sd_mean[eac] = gaussian_discr(df1,eac)
"""

k_outs_ge_50k_p = []
k_outs_le_50k_p = []

k_outs_ge_50k_p,k_outs_le_50k_p = cross_valid(df,10)

#If you want to provide your own test set then un-comment the below function and add fipe path to your test set.
#df_test = pd.read_csv('you_file_path.csv')
#y_agg_pred_avg = diff_test_set(df_test,k_outs_ge_50k_p,k_outs_le_50k_p)

#Run below function after calculating above to get accuracy results.
#simple_test_acc(y_agg_pred_avg,df_test['Output'])
