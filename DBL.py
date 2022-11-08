# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:51:55 2020
baseline correction
@author: Xue Fang
"""

import numpy as np
import pandas as pd
import os

def de_baseline(df, region=(750, 800)):
    left, right= region
    bias = df.absorb[ (df.lam>=left) & (df.lam<=right) ].mean()
    df.absorb = df.absorb - bias
    return df

def read_file(filename):
    df = pd.read_csv(filename, skiprows=2, names=['lam','absorb'], sep='\t')
    return df

def parse_name(filename):
    file=filename.split('\\')[-1]
    name=file.split('.')[0]
    return name

def dbl_all(path):
    all_dbl_file={}
    for file in os.listdir(path):
        if file[-4:]=='.txt':
            df=de_baseline(read_file(path+'\\'+file))
            name=parse_name(file)
            all_dbl_file['Wavelength']=df.lam
            all_dbl_file[name+'_DBL']=df.absorb
    dbl_df=pd.DataFrame(all_dbl_file)
    dbl_df.to_excel(path+'\\DBL_'+path.split('\\')[-1]+'.xlsx', index=None)
            
def to_excel(path):
   all_to_excel_file={}
   for file in os.listdir(path):
       if file[-4:]=='.txt':
           df=read_file(path+'\\'+file)
           name=parse_name(file)
           name_with_dot=name.split('_')[0]+'_'+add_dot(name.split('_')[-1])
           all_to_excel_file['Wavelength']=df.lam
           all_to_excel_file[name_with_dot]=df.absorb
   to_excel_df=pd.DataFrame(all_to_excel_file)
   to_excel_df=to_excel_df[(to_excel_df.Wavelength>=400)&(to_excel_df.Wavelength<=800)]
   to_excel_df.to_excel(path+'\\'+path.split('\\')[-1]+'.xlsx',index=None)
    
def read_excel(excelname):
    dfe= pd.read_excel(excelname, index=None)
    return dfe

def dbl_excel(excelname):
    dfe=read_excel(excelname)
    for i,x in dfe.iteritems():
        if i !='Wavelength':
            dfe[i][:]=dfe[i][:]-dfe[i][701:801].mean()
    dfe.to_excel(excelname+'_DBL.xlsx', index=None)
    
def max_peak_td(excelname):
    dfe=read_excel(excelname)
    left_0_0=510
    right_0_0=530
    left_0_1=460
    right_0_1=490
    left_0_2=430
    right_0_2=450
    df_absorb_col=['Time',
               'Lam_0_0',
               'Abs_0_0',
               'Lam_0_1',
               'Abs_0_1',
               'Lam_0_2',
               'Abs_0_2',
               ]
    df_absorb=pd.DataFrame(columns=df_absorb_col)
    for i,x in dfe.iteritems():
        if i !='Wavelength' and i != 'DMF_SON':
            
            y_0_0=dfe[(dfe.Wavelength>=left_0_0)&(dfe.Wavelength<=right_0_0)][i]
            A_0_0=y_0_0.max()
            lam_0_0=dfe.Wavelength[y_0_0.idxmax()]
            y_0_1=dfe[(dfe.Wavelength>=left_0_1)&(dfe.Wavelength<=right_0_1)][i]
            A_0_1=y_0_1.max()
            lam_0_1=dfe.Wavelength[y_0_1.idxmax()]
            y_0_2=dfe[(dfe.Wavelength>=left_0_2)&(dfe.Wavelength<=right_0_2)][i]
            A_0_2=y_0_2.max()
            lam_0_2=dfe.Wavelength[y_0_2.idxmax()]
        
  
            df_stack=pd.DataFrame({'Time':[int(i)],
                                   'Lam_0_0':[lam_0_0],
                                   'Abs_0_0':[A_0_0],
                                   'Lam_0_1':[lam_0_1],
                                   'Abs_0_1':[A_0_1],
                                   'Lam_0_2':[lam_0_2],
                                   'Abs_0_2':[A_0_2],
                                   })
            df_absorb=df_absorb.append(df_stack,ignore_index=True)
    
    df_absorb['A_0_0/A_0_1']=df_absorb['Abs_0_0']/df_absorb['Abs_0_1']
    df_absorb['A_0_0/A_0_2']=df_absorb['Abs_0_0']/df_absorb['Abs_0_2']
    df_absorb.to_excel(excelname+'peak_ratio.xlsx',index=None)
   # print (df_absorb)

def add_dot(number):
    int_part=number.split('E')[0]
    if len(int_part)!=1:
        int_dot_dec=int_part[0]+'.'+int_part[1:]
    else:
        int_dot_dec=int_part+'.0'
    return int_dot_dec+'E'+number.split('E')[-1]                                
                      
def max_peak_conc(excelname):
    dfe=read_excel(excelname)
    left_0_0=510
    right_0_0=535
    left_0_1=460
    right_0_1=500
    left_0_2=430
    right_0_2=450
    df_absorb_col=['Conc',
               'Lam_0_0',
               'Abs_0_0',
               'Lam_0_1',
               'Abs_0_1',
               'Lam_0_2',
               'Abs_0_2',
               ]
    df_absorb=pd.DataFrame(columns=df_absorb_col)
    for i,x in dfe.iteritems():
        if i !='Wavelength':
            
            y_0_0=dfe[(dfe.Wavelength>=left_0_0)&(dfe.Wavelength<=right_0_0)][i]
            A_0_0=y_0_0.max()
            lam_0_0=dfe.Wavelength[y_0_0.idxmax()]
            y_0_1=dfe[(dfe.Wavelength>=left_0_1)&(dfe.Wavelength<=right_0_1)][i]
            A_0_1=y_0_1.max()
            lam_0_1=dfe.Wavelength[y_0_1.idxmax()]
            y_0_2=dfe[(dfe.Wavelength>=left_0_2)&(dfe.Wavelength<=right_0_2)][i]
            A_0_2=y_0_2.max()
            lam_0_2=dfe.Wavelength[y_0_2.idxmax()]
            #conc=add_dot(i.split('_')[-1])
            conc=i.split('_')[-1]
  
            df_stack=pd.DataFrame({'Conc':[conc],
                                   'Lam_0_0':[lam_0_0],
                                   'Abs_0_0':[A_0_0],
                                   'Lam_0_1':[lam_0_1],
                                   'Abs_0_1':[A_0_1],
                                   'Lam_0_2':[lam_0_2],
                                   'Abs_0_2':[A_0_2],
                                   })
            df_absorb=df_absorb.append(df_stack,ignore_index=True)
    
    df_absorb['A_0_0/A_0_1']=df_absorb['Abs_0_0']/df_absorb['Abs_0_1']
    df_absorb['A_0_0/A_0_2']=df_absorb['Abs_0_0']/df_absorb['Abs_0_2']
    df_absorb.to_excel(excelname+'peak_ratio.xlsx',index=None)
   # print (df_absorb)
    
                
def global_to_excel(path):
   all_to_excel_file={}
   for file in os.listdir(path):
       if file[-4:]=='.txt':
           df=read_file(path+'\\'+file)
           name=parse_name(file)
           name_with_dot=name.split('_')[0]+'_'+add_dot(name.split('_')[-1])
           all_to_excel_file['Wavelength']=df.lam
           all_to_excel_file[name_with_dot]=df.absorb
   to_excel_df=pd.DataFrame(all_to_excel_file)
   to_excel_df.to_excel(path+'\\'+path.split('\\')[-1]+'.xlsx',index=None)
    #return wavelength

