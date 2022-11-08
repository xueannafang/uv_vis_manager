# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:19:32 2020

@author: Xue Fang
"""
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelmax

class UVVis:
    '''
    The object of this class is uv-vis raw data txt file
    '''
    
    def __init__(self,path,sep='\t',is_temp=0):
        '''
        This path includes filename.
        '''
        self.path = path
        self.data = self.read_file(path, sep)
        self.data = self.data[(self.data.lam>=400) & (self.data.lam<=800)].astype('float64')
        if is_temp==0:
            self.parse_name(path)
        elif is_temp==1:
            self.parse_name_temp(path)
    
    def read_file(self,path, sep):
        '''
        This file is to read files. data from UV2600 is default. data from sop should change sep to ','.
        '''
        if sep=='\t':
            skiprows=2
        elif sep==',':
            skiprows=1
        df = pd.read_csv(path, skiprows=skiprows, names=['lam','absorb'], sep=sep)
        return df
    
    def parse_name(self,path):
        '''
        This function is to extract info of molecule name, solvent name, concentration (with dot), raw concentration (number) from conc-dependent data.
        It also determined the optical path of cuvette according to the file name.
        '''
        file = path.split('\\')[-1]
        solv_folder = path.split('\\')[-2]
        self.name = file.split('.')[0]
        self.number = self.name.split('_')[-1]
        self.solvent = self.name.split('_')[-2]
        self.conc=self.add_dot(self.number) #M
        self.opt_path = 1 #cm
        if solv_folder.split('_')[-1] == 'l':
            self.opt_path = 0.4 #cm
    
    def parse_name_temp(self, path):
        '''
        This function is to extract info of solvent name, concentration (with dot), raw concentration (number) and temperature from temp-dependent data.
        '''
        filename = path.split('\\')[-1]
        solv_folder = path.split('\\')[-2]
        self.name = filename.split('.')[0]
        if filename.split('.')[-3] == 'Sample':
            self.number = self.name.split('_')[-2]
            self.solvent = self.name.split('_')[-3]
            self.conc = self.add_dot(self.number)
            self.temp = float(self.name.split('_')[-1])+273
        else:
            self.temp=0 #0 means background
    
    def add_dot(self,number):
        '''
        This function is to add decimal point between integer and decimals, and to create the scientific expression.
        '''
        int_part = number.split('E')[0]
        if len(int_part)!=1:
            int_dot_dec = int_part[0]+'.'+int_part[1:]
        else:
            int_dot_dec = int_part+'.0'
        number_with_dot = int_dot_dec+'E'+number.split('E')[-1]
        return float(number_with_dot)

    def savgol_smoothing(self, win_len=29, poly_order=2, **kwargs):
        """
        Perform Savitzky-Golay smoothing algorithm. (window rolling polynomial regression)
        """
        self.data['smooth'] = savgol_filter(self.data.absorb, win_len, poly_order, **kwargs)
        #self.compare_plot(f'Sacitzky-Golay Smoothing with window length {win_len} and polynomial order {poly_order}')

    def calc_eps(self,norm_700=0,smooth=0,**kwargs):
        '''
        This step is to calculate epsilon.
        '''
        self.savgol_smoothing(**kwargs)
        if norm_700==0:
            if smooth==0:
                self.data['eps'] = self.data.absorb/(float(self.conc)*self.opt_path)
            elif smooth==1:
                self.data['eps'] = self.data.smooth/(float(self.conc)*self.opt_path)
        elif norm_700==1:  
            if smooth==0:
                self.data['eps'] = (self.data.absorb-self.data['absorb'][self.data['lam']==700].values)/(float(self.conc)*self.opt_path)
            elif smooth==1:
                self.data['eps'] = (self.data.smooth-self.data['smooth'][self.data['lam']==700].values)/(float(self.conc)*self.opt_path)
            avg_700=self.data[(self.data['lam']>=700) & (self.data['lam']<=800)]['eps'].mean()
            self.data['eps']-=avg_700

    def find_eps_peak(self, correction, left_0_0=510,right_0_0=530,left_0_1=460,right_0_1=495,left_0_2=430,right_0_2=450,left_agg=550,right_agg=600):
        E_0_0=self.data[(self.data.lam<=right_0_0)&(self.data.lam>=left_0_0)].eps.max()
        L_0_0=self.data['lam'][self.data[(self.data.lam<=right_0_0)&(self.data.lam>=left_0_0)]['eps'].idxmax()]
        E_0_1=self.data[(self.data.lam<=right_0_1)&(self.data.lam>=left_0_1)].eps.max()
        L_0_1=self.data['lam'][self.data[(self.data.lam<=right_0_1)&(self.data.lam>=left_0_1)]['eps'].idxmax()]
        
        df=self.data.copy()
        df['eps']=df['eps']-correction
        
        E_agg_r=df[(df.lam<=right_agg)&(df.lam>=left_agg)].eps.max()
        L_agg=df['lam'][df[(df.lam<=right_agg)&(df.lam>=left_agg)]['eps'].idxmax()]
        
        self.E_agg=self.data['eps'][self.data[self.data.lam==int(L_agg)].index].values
        #self.A_agg_norm=self.A_agg-self.data['eps'][ self.data['lam']==700 ].values

        self.E_0_0=(E_0_0,L_0_0)
        self.E_0_1=(E_0_1,L_0_1)
        self.E_agg_r=(E_agg_r,L_agg)
        self.RE01=float(E_0_0)/float(E_0_1)
        
    def find_abs_peak(self, L_agg=560, left_0_0=510,right_0_0=530,left_0_1=460,right_0_1=495,left_0_2=430,right_0_2=450,left_agg=550,right_agg=600):
        '''
        This function is to find the max absorbance peaks of conc-dependent uvvis according to the raw txt.
        '''
        A_0_0=self.data[(self.data.lam<=right_0_0)&(self.data.lam>=left_0_0)].absorb.max()
        L_0_0=self.data['lam'][self.data[(self.data.lam<=right_0_0)&(self.data.lam>=left_0_0)]['absorb'].idxmax()]
        A_0_1=self.data[(self.data.lam<=right_0_1)&(self.data.lam>=left_0_1)].absorb.max()
        L_0_1=self.data['lam'][self.data[(self.data.lam<=right_0_1)&(self.data.lam>=left_0_1)]['absorb'].idxmax()]
        #NA_0_0=A_0_0-self.data['absorb'][ self.data['lam']==700 ].values
        #NA_0_1=A_0_1-self.data['absorb'][ self.data['lam']==700 ].values
        
        #df=self.data.copy()
        #df['absorb']=df['absorb']-correction
        
        #A_agg_r=df[(df.lam<=right_agg)&(df.lam>=left_agg)].absorb.max()
        #L_agg=df['lam'][df[(df.lam<=right_agg)&(df.lam>=left_agg)]['absorb'].idxmax()]
        
        A_agg=self.data['absorb'][self.data[self.data.lam==int(L_agg)].index].values
        #self.A_agg_norm=self.A_agg-self.data['absorb'][ self.data['lam']==700 ].values

        self.A_0_0=(A_0_0,L_0_0)
        self.A_0_1=(A_0_1,L_0_1)
        #self.A_agg_r=(A_agg_r50,L_agg)
        self.A_agg=(float(A_agg),L_agg)
        self.A_agg_norm=float(float(A_agg)/self.conc)
        self.RA01=float(A_0_0)/float(A_0_1)
        #self.NA01=float(NA_0_0)/float(NA_0_1)

    def max_peak_temp(self, correction, left_0_0=510,right_0_0=530,left_0_1=460,right_0_1=495,left_0_2=430,right_0_2=450,left_agg=550,right_agg=600):
        '''
        This function is to find the max absorbance peaks of temp-dependent uvvis according to the raw rcf.
        '''
        A_0_0=self.data[(self.data.lam<=right_0_0)&(self.data.lam>=left_0_0)].absorb.max()
        L_0_0=self.data['lam'][self.data[(self.data.lam<=right_0_0)&(self.data.lam>=left_0_0)]['absorb'].idxmax()]
        A_0_1=self.data[(self.data.lam<=right_0_1)&(self.data.lam>=left_0_1)].absorb.max()
        L_0_1=self.data['lam'][self.data[(self.data.lam<=right_0_1)&(self.data.lam>=left_0_1)]['absorb'].idxmax()]
        NA_0_0=A_0_0-self.data['absorb'][ self.data['lam']==700 ].values
        NA_0_1=A_0_1-self.data['absorb'][ self.data['lam']==700 ].values
        
        df=self.data.copy()
        df['absorb']=df['absorb']-correction
        
        A_agg_r50=df[(df.lam<=right_agg)&(df.lam>=left_agg)].absorb.max()
        L_agg=df['lam'][df[(df.lam<=right_agg)&(df.lam>=left_agg)]['absorb'].idxmax()]
        
        self.A_agg=self.data['absorb'][self.data[self.data.lam==int(L_agg)].index].values
        self.A_agg_norm=self.A_agg-self.data['absorb'][ self.data['lam']==700 ].values

        self.A_0_0=(A_0_0,L_0_0)
        self.A_0_1=(A_0_1,L_0_1)
        self.A_agg_r50=(A_agg_r50,L_agg)
        self.RA01=float(A_0_0)/float(A_0_1)
        self.NA01=float(NA_0_0)/float(NA_0_1)
        
        

class UVVisManager:
    '''
    Find the file.
    '''
    def __init__(self, father_path):
        self.father_path = father_path
        self._all_txt()
        self._all_excel()
        self._all_rcf()
    
    def _all_txt(self):
        '''
        This function is for finding all raw data txt files from uvvis2600, w620
        '''
        self.all_txt = {k : [file for file in os.listdir(self.father_path+'\\'+k) 
                            if file[-4:] == '.txt'] 
                        for k in os.listdir(self.father_path) 
                        if os.path.isdir(self.father_path+'\\'+k)}
    
    def _all_excel(self):
        '''
        This function is for finding all system-generated excel files from uvvis2600,w620
        '''
        self.all_excel = {k : [file for file in os.listdir(self.father_path+'\\'+k)
                            if file[-5:] == '.xlsx']
                        for k in os.listdir(self.father_path) 
                        if os.path.isdir(self.father_path+'\\'+k)}
        
    def _all_rcf(self):
        '''
        This function is for finding all system-generated excel files from lambda35, sop1.38. 
        rcf means files ending with raw.csv
        '''
        self.all_rcf = {k : [file for file in os.listdir(self.father_path+'\\'+k)
                            if file[-7:] == 'Raw.csv']
                        for k in os.listdir(self.father_path) 
                        if os.path.isdir(self.father_path+'\\'+k)}
        
    def eps_excel(self,norm_700,conc_limit = 1.0E-5):
        '''
        This function is to transfer calculated epsilon (molar absorbance) and corresponding wavelength to excel.
        '''
        for folder in self.all_txt:
            solv_df = pd.DataFrame([])
            uvvis_list = []
            for txt in self.all_txt[folder]:
                path='\\'.join([self.father_path,folder,txt])
                uvvis=UVVis(path)
                if uvvis.conc <= conc_limit:
                    smooth=1
                else:
                    smooth=0
                uvvis.calc_eps(norm_700,smooth)
                uvvis_list.append(uvvis)
            print('uvvis_list generation successfully')
            uvvis_list.sort(key=lambda x: x.conc)
            print('uvvis_list sort successfully')
            for i, uvvis in enumerate(uvvis_list):
                if i == 0:
                    solv_df['Wavelength'] = uvvis.data.lam
                solv_df['{:.1E}'.format(uvvis.conc)] = uvvis.data.eps
            #ascending column
            if norm_700==0 and smooth==0:
                solv_df.to_excel('\\'.join([self.father_path,folder,folder])+'_eps_'+'.xlsx',
                             index=None)
            elif norm_700==1 and smooth==0:
                solv_df.to_excel('\\'.join([self.father_path,folder,folder])+'_eps_norm_'+'.xlsx',
                             index=None)
            elif norm_700==1 and smooth==1:
                solv_df.to_excel('\\'.join([self.father_path,folder,folder])+'_eps_norm_smooth'+'.xlsx',
                             index=None)
            print(f'{folder} clear') #To suggest which folder has been treated.
    
    def txt_excel(self):
        '''
        This function is to transfer absorbance and corresponding wavelength in the raw txt data file to excel.
        '''
        for folder in self.all_txt:
            solv_df = pd.DataFrame([])
            uvvis_list = []
            for txt in self.all_txt[folder]:
                path='\\'.join([self.father_path,folder,txt])
                uvvis=UVVis(path)
                #uvvis.calc_eps()
                uvvis_list.append(uvvis)
            print('uvvis_list generation successfully')
            uvvis_list.sort(key=lambda x: x.conc)
            print('uvvis_list sort successfully')
            for i, uvvis in enumerate(uvvis_list):
                if i == 0:
                    solv_df['Wavelength'] = uvvis.data.lam
                solv_df['{:.1E}'.format(uvvis.conc)] = uvvis.data.absorb
            #ascending column
            solv_df.to_excel('\\'.join([self.father_path,folder,folder])+'_abs_'+'.xlsx',
                             index=None)
            print(f'{folder} clear')
    
    def rcf_excel(self):
        '''
        This function is to transfer raw csv data to a combined excel.
        '''
        for folder in self.all_rcf:
            solv_df = pd.DataFrame([])
            uvvis_list = []
            for csv in self.all_rcf[folder]:
                path = '\\'.join([self.father_path,folder,csv])
                print(f'{path} start loaded')
                uvvis=UVVis(path,sep=',',is_temp=1)
                uvvis_list.append(uvvis)
                print(f'{path} success')
            uvvis_list.sort(key=lambda x:x.temp)
            for i, uvvis in enumerate(uvvis_list):
                if i==0:
                    solv_df['Wavelength']=uvvis.data.lam
                solv_df[str(uvvis.temp)] = uvvis.data.absorb
            solv_df.to_excel('\\'.join([self.father_path,folder,folder+'_'+str('{:.1E}'.format(uvvis.conc))])+'temp'+'.xlsx',
                             index=None)
            
    def rcf_norm_excel(self):
        '''
        This function is to normalize all the intensity of rcf data to 0 (at 700 nm)
        '''
        for folder in self.all_rcf:
            solv_df = pd.DataFrame([])
            uvvis_list = []
            for csv in self.all_rcf[folder]:
                path = '\\'.join([self.father_path,folder,csv])
                print(f'{path} start loaded')
                uvvis=UVVis(path,sep=',',is_temp=1)
                uvvis_list.append(uvvis)
                print(f'{path} success')
            uvvis_list.sort(key=lambda x:x.temp)
            for i, uvvis in enumerate(uvvis_list):
                if i==0:
                    solv_df['Wavelength']=uvvis.data.lam
                solv_df[str(uvvis.temp)] = uvvis.data.absorb-uvvis.data['absorb'][ uvvis.data['lam']==700 ].values
            solv_df.to_excel('\\'.join([self.father_path,folder,folder+'_'+str('{:.1E}'.format(uvvis.conc))])+'temp_norm'+'.xlsx',
                             index=None)
            
    def uvvis_class_fdict(self,is_temp=0):
        '''
        This function is to build a dictionary for uvvis info from temp-UV (rcf), 
        key=folder name
        value=UVVis (class)
        '''
        self.uvvis_class_dict={}
        if is_temp==1:
            for folder in self.all_rcf:
                uvvis_list=[]
                for csv in self.all_rcf[folder]:
                    path = '\\'.join([self.father_path,folder,csv])

                    uvvis_list.append(UVVis(path,sep=','))
                self.uvvis_class_dict[folder]=uvvis_list
        elif is_temp==0:
            for folder in self.all_txt:
                uvvis_list=[]
                for txt in self.all_txt[folder]:
                    path = '\\'.join([self.father_path,folder,txt])
                    uvvis= UVVis(path)
                    if uvvis.conc<=1.0E-5:
                        smooth=1
                    else:
                        smooth=0
                    uvvis.calc_eps(norm_700=1,smooth=smooth)
                    uvvis_list.append(uvvis)
                self.uvvis_class_dict[folder]=uvvis_list
                    
                
        #return uvvis_class_dict
    
    def max_peak_temp_excel(self,folder,**kwargs):
        df_absorb_col=['Temp',
               'Lam_0_0',
               'Abs_0_0',
               'Lam_0_1',
               'Abs_0_1',
               'Lam_agg',
               'Abs_agg_r50',
               'A_0_0/A_0_1',
               'NA_0_0/NA_0_1',
               'A_agg_norm',
               ]
        df_absorb=pd.DataFrame(columns=df_absorb_col)
        
        self.uvvis_class_fdict(is_temp=1)
        #print(self.uvvis_class_dict[folder].sort(key=lambda x:x.temp))
        
        correction=sorted(self.uvvis_class_dict[folder],key=lambda x:x.temp)[-1].data.absorb
        for uvvis_class in self.uvvis_class_dict[folder]:
            uvvis_class.max_peak_temp(correction, **kwargs)
            df_stack=pd.DataFrame({'Temp':[uvvis_class.temp],
                                   'Lam_0_0':[uvvis_class.A_0_0[-1]],
                                   'Abs_0_0':[uvvis_class.A_0_0[0]],
                                   'Lam_0_1':[uvvis_class.A_0_1[-1]],
                                   'Abs_0_1':[uvvis_class.A_0_1[0]],
                                   'Lam_agg':[uvvis_class.A_agg_r50[-1]],
                                   'Abs_agg_r50':[uvvis_class.A_agg_r50[0]],
                                   'Abs_agg':[uvvis_class.A_agg[0]],
                                   'A_0_0/A_0_1':[uvvis_class.RA01],
                                   'NA_0_0/NA_0_1':[uvvis_class.NA01],
                                   'A_agg_norm':[uvvis_class.A_agg_norm[0]]
                                   })
            df_absorb=df_absorb.append(df_stack,ignore_index=True)
            
        df_absorb = df_absorb[df_absorb.columns[::-1]]
        df_absorb.to_excel('\\'.join([self.father_path,folder,folder+'_'])+'temp_AR'+'.xlsx',
                             index=None)

    def max_peak_excel(self, task='eps',**kwargs):
        if task == 'eps':
            df_col=['Conc',
               'Lam_0_0',
               'Eps_0_0',
               'Lam_0_1',
               'Eps_0_1',
               'Lam_agg',
               'Eps_agg_r',
               'Eps_agg',
               'E_0_0/E_0_1',
               ]
        
        elif task == 'abs':
            df_col=['Conc',
               'Lam_0_0',
               'Abs_0_0',
               'Lam_0_1',
               'Abs_0_1',
               'Lam_agg',
               'Abs_agg',
               'Abs_conc_norm',
               'A_0_0/A_0_1',
               ]
        
        df_absorb=pd.DataFrame(columns=df_col)
        self.uvvis_class_fdict(is_temp=0)
        #print(self.uvvis_class_dict[folder].sort(key=lambda x:x.conc))
        
        for folder in self.uvvis_class_dict:
            df_absorb=pd.DataFrame([])
            if task=='eps':
                correction = sorted(self.uvvis_class_dict[folder],key=lambda x:x.conc)[0].data['eps']
                for uvvis_class in self.uvvis_class_dict[folder]:
                    if folder=='DMF':
                        uvvis_class.find_eps_peak(correction, left_agg=567, right_agg=583,**kwargs)
                    else:
                        uvvis_class.find_eps_peak(correction, **kwargs)
                    
                    df_stack=pd.DataFrame({'Conc':[uvvis_class.conc],
                                    'Lam_0_0':[uvvis_class.E_0_0[-1]],
                                    'Eps_0_0':[uvvis_class.E_0_0[0]],
                                    'Lam_0_1':[uvvis_class.E_0_1[-1]],
                                    'Eps_0_1':[uvvis_class.E_0_1[0]],
                                    'Lam_agg':[uvvis_class.E_agg_r[-1]],
                                    'Eps_agg_r':[uvvis_class.E_agg_r[0]],
                                    'Eps_agg':[uvvis_class.E_agg[0]],
                                    'E_0_0/E_0_1':[uvvis_class.RE01],
                                    })
                    df_absorb=df_absorb.append(df_stack,ignore_index=True)
                df_absorb = df_absorb[df_absorb.columns].sort_values(by=['Conc'])
                df_absorb.to_excel('\\'.join([self.father_path,folder,folder+'_'])+'EPS_AR'+'.xlsx',
                                index=None)
                print(folder +' eps ratio calculated succcessfully.')
            elif task=='abs':
                for uvvis_class in self.uvvis_class_dict[folder]:
                    uvvis_class.find_abs_peak(**kwargs)
                    df_stack=pd.DataFrame({'Conc':[uvvis_class.conc],
                                    'Lam_0_0':[uvvis_class.A_0_0[-1]],
                                    'Abs_0_0':[uvvis_class.A_0_0[0]],
                                    'Lam_0_1':[uvvis_class.A_0_1[-1]],
                                    'Abs_0_1':[uvvis_class.A_0_1[0]],
                                    'Lam_agg':[uvvis_class.A_agg[-1]],
                                    'Abs_agg':[uvvis_class.A_agg[0]],
                                    'Abs_agg_conc_norm':[uvvis_class.A_agg_norm],
                                    'A_0_0/A_0_1':[uvvis_class.RA01],
                                    })
                    df_absorb=df_absorb.append(df_stack,ignore_index=True)
                df_absorb = df_absorb[df_absorb.columns].sort_values(by=['Conc'])
                df_absorb.to_excel('\\'.join([self.father_path,folder,folder+'_'])+'Abs_AR'+'.xlsx',
                                index=None)
                print(folder +' abs ratio calculated succcessfully.')

    def eps_relative(self):
        self.uvvis_class_fdict(is_temp=0)
        #print(self.uvvis_class_dict[folder].sort(key=lambda x:x.conc))
        
        for folder in self.uvvis_class_dict:
            self.uvvis_class_dict[folder].sort(key=lambda x: x.conc)
            correction = self.uvvis_class_dict[folder][0].data['eps']
            corr_df=pd.DataFrame([])
            for uvvis_class in self.uvvis_class_dict[folder]:
                eps_copy=uvvis_class.data['eps'].copy()
                eps_copy=uvvis_class.data['eps']-correction
                corr_df['lam']=uvvis_class.data['lam']
                corr_df['{:.1E}'.format(uvvis_class.conc)] = eps_copy
            
            corr_df.to_excel('\\'.join([self.father_path,folder,folder+'_'])+'EPS_rel'+'.xlsx',
                             index=None)