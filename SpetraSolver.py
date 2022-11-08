# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:55:19 2020

@author: Xue Fang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelmax

test_filename = 'PDI-BA-1-0319/0319-PDI-BA-1-DMSO/0319-PDI-BA-1-DMSO-025E-4.txt'

def gaussian(x, amp, cen, sigma):
    return amp/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-cen)/sigma)**2)

def polynomial(x, n, *args):
    result = 0
    if len(args) == n:
        for i,c in enumerate(args):
            result += c*x**i
        return result
    else:
        raise ValueError('Not enough coefficient!')
        
def exponential(x, a, k, c):
    return a*np.exp(k*x)+c
            
def n_gaussian(x, n, *args):
    g = 0
    for i in range(n):
        p = args[3*i:3*(i+1)]
        g += gaussian(x, *p)
    return g

def fft_filter(y, cutoff):
    yfft = np.fft.rfft(y)
    yfft_new = yfft.copy()
    if isinstance(cutoff, (float, int)):
        if cutoff < 1:
            N_cut = int((1-cutoff)*len(yfft))        
            yfft_new[N_cut:]=0
        elif cutoff >= 1:
            yfft_new[cutoff] = 0       
    elif len(cutoff) == 2:
        if cutoff[0] < 1:
            start=int((1-cutoff[0])*len(yfft))
        elif cutoff[0] >=1:
            start = cutoff[0]
        if cutoff[1] < 1:
            end=int((1-cutoff[0])*len(yfft))
        elif cutoff[1] >=1:
            end = cutoff[1]
        yfft_new[start:end]=0
    y_new = np.fft.irfft(yfft_new, len(y))
    return y, yfft, yfft_new, y_new


class Spectra: 
    def __init__(self, filename, name=None):
        self.df = pd.read_csv(filename, skiprows=2, names=['lam','absorb'], sep='\t')
        self.lam = self.df.lam.values
        self.absorb = self.df.absorb.values        
        self.cache = self.absorb
        if name:
            self.name = name
#        self.baseline_methods = {
#            'average': self.average_baseline,
#        }
#     
#        self.smoothing_methods = {
#            'fft': self.fft_smoothing,
#            'savgol': self.savgol_smoothing,
#        }
#        
#        self.peak_compare_methods = {
#            'norm': lambda x,y: (x-y)/(x+y),
#            'ratio': lambda x,y : x/y        
#        }

    


    
    def average_baseline(self, region=(750, 800)):
        left, right= region
        bias = self.cache[ (self.df.lam>=left) & (self.df.lam<=right) ].mean()
        self.cache = self.cache - bias
        self.compare_plot(f'Baseline settings for bias {bias}')
        return bias
    
    def fft_smoothing(self, cut_off=0.1):
        y, yfft, yfft_new, y_new = fft_filter(self.cache, cut_off)        
        self.cache = y_new
        self.fft_plot(y,yfft,yfft_new,y_new,cut_off)
        self.compare_plot(f'FFT Smoothing for cut off = {cut_off}')
        return y, yfft, yfft_new, y_new
    
    def savgol_smoothing(self, win_len=15, poly_order=2, **kwargs):
        """
        Perform Savitzky-Golay smoothing algorithm. (window rolling polynomial regression)
        """
        self.cache = savgol_filter(self.cache, win_len, poly_order, **kwargs)
        self.compare_plot(f'Sacitzky-Golay Smoothing with window length {win_len} and polynomial order {poly_order}')
        
        
    def init_guess(self, n, bound=(400,650), p0=[]):
        """
        bound gives the range of wavelength
        p0 is a list of float or tuple with 3 elements(amp, center, sigma)
        """
        locs = np.linspace(bound[0], bound[1], n)
        amps = np.ones(n)
        sigmas = np.ones(n)
        dx = (bound[1]-bound[0])/(n-1)
        if len(p0) != 0:
            for j, p in enumerate(p0):
                if isinstance(p, (int, float)):
                    for i,x in enumerate(locs):
                        if abs(p-x)<=dx/2:
                            locs[i] = p
                            break 
                elif len(p) == 3:
                     for i,x in enumerate(locs):
                        if abs(p[1]-x)<=dx/2:
                            amps[i] = p[0]
                            locs[i] = p[1]
                            sigmas[i] = p[2]
                            break 
                else:
                    raise ValueError('Invalid input for p0!')
        self.terms = n
        self.p_group = np.array([amps, locs, sigmas]).T
        self.p_fit = self.p_group.reshape(3*n)
    
    def _get_p_bound(self):
        low_bounds = np.zeros(3*self.terms)
        high_bounds = np.zeros(3*self.terms)+np.inf
        for i in range(3*self.terms):
            if i%3 == 2:
                dx = (self.lam.max()-self.lam.min())/self.terms
                low_bounds[i]=1
                high_bounds[i]=0.66*dx
            elif i%3 == 1:
                low_bounds[i] = self.lam.min()-50
                high_bounds[i] = self.lam.max()
        return low_bounds, high_bounds
        
    def fit_n(self, n, p0=[], bound=(400, 650), text=''):
        self.init_guess(n, bound=bound, p0=p0)
        def g_func(x, *args):
            return n_gaussian(x, n, *args)
        self.p_fit, self.cov = curve_fit(g_func, self.lam, self.cache, p0=self.p_fit, bounds=self._get_p_bound())
        self.error = np.mean((self.cache-g_func(self.lam, *self.p_fit))**2)
        self.p_group = self.p_fit.reshape(n,3)
        self.peaks = {
            i: {
                'cen': p[1],
                'max': gaussian(p[1], *p),
                'amp': p[0],
                'sigma': p[2],
            }
            for i,p in enumerate(self.p_group)
        }
        self.g_total = lambda x: g_func(x, *self.p_fit)
        self.g_group = [gaussian(self.lam, *p)for p in self.p_group]
        #self.gaussian_plot(title=text)
        self.no_exp_plot(title=text)
        print(self.error)
                    
    def _tr(self, index1, index2):
        """
        norm option returns index1-index2/index1+index2
        ratio option retruns index1/index2
        """
        A = self.peaks[index1]
        B = self.peaks[index2]
        return (A['cen'],A['max']), (B['cen'],B['max'])
    
    def _ir(self, index1, index2):
        y = self.g_total(self.lam)
        maximas = argrelmax(y)[0]
        loc = (self.peaks[index1]['cen']+self.peaks[index2]['cen'])/2
        for i,pos in enumerate(maximas):
            if loc < self.lam[pos]:
                right = pos
                left = maximas[i-1]
                break
        return (self.lam[right], y[right]), (self.lam[left], y[left])
    
    def peak_describe(self, index1, index2):
        A, B = self._tr(index1, index2)
        ntr = (A[1]-B[1])/(A[1]+B[1])
        atr = A[1]/B[1]
        A_, B_ = self._ir(index1, index2)
        nir = (A_[1]-B_[1])/(A_[1]+B_[1])
        air = A_[1]/B_[1]
        TR = {
            'lam': (A[0], B[0]),
            'abs': (A[1], B[1]),
            'ntr': ntr,
            'atr': atr,
        }
        IR = {
            'lam': (A_[0], B_[0]),
            'abs': (A_[1], B_[1]),
            'nir': nir,
            'air': air,
        }
        report = {'TR':TR, 'IR':IR}
        return report

    def plot(self):
        plt.plot(self.lam, self.absorb)
        plt.show()
        
    def compare_plot(self, text):
        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.plot(self.lam, self.absorb, 'r--', lw=2, label='Original data')
        ax.plot(self.lam, self.cache, 'k', alpha=0.7, lw=2.5, label='New data')
        ax.set_title(text)
        ax.legend()
        plt.show()
        
    def fft_plot(self, y, yfft, yfft_new, y_new, cut_off):
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        ax[0,0].plot(self.lam, y)
        ax[0,0].set_title('Original Signal')
        ax[0,1].plot(np.real(yfft))
        ax[0,1].plot(np.imag(yfft))
        ax[0,1].set_title('FFT Signal')
        ax[1,0].plot(np.real(yfft_new))
        ax[1,0].plot(np.imag(yfft_new))
        ax[1,0].set_title(f'FFT with cutoff {cut_off}')
        ax[1,1].plot(self.lam, y_new,'-', lw=2, label='Smoothed')
        ax[1,1].plot(self.lam, y,'--',alpha=0.5, label='Original')
        ax[1,1].set_title('Smoothed Signal')
        ax[1,1].legend()
        if 'name' in self.__dict__:
            fig.suptitle(f'FFT Smoothing for {self.name}')
        else:
            fig.suptitle('FFT Smoothing')
        plt.show()
        
    def gaussian_plot(self, title=''):
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        ax.plot(self.lam, self.g_total(self.lam), c='r', lw=2, label='Fit curve')
        ax.plot(self.lam, self.absorb, 'gray',alpha=0.6, label='Experimental data')
        ax.plot(self.lam, self.cache, 'b',alpha=0.6, label='Smoothed data')
        for gn in self.g_group:
            ax.plot(self.lam, gn, alpha=0.6)
            ax.fill_between(self.lam, gn, 0, alpha=0.6)
        ax.legend(loc=0,fontsize=20)
        ax.set_xlabel('Wavelength (nm)',fontsize=20)
        ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
        ax.set_ylim(0,self.absorb.max()*1.1)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        for p in self.peaks.values():
            if self.lam.min() <= p['cen'] <= self.lam.max():
                ax.vlines(p['cen'],0,p['max'], linestyles='dashed', lw=0.5)
        if title:
            ax.set_title(title, fontsize=20)
        plt.show()
            
    def peaks_plot(self, indicies, title=''):
        g = 0
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        for i in indicies:
            g += self.g_group[i]
            ax.plot(self.lam, self.g_group[i], alpha=0.6)
            ax.fill_between(self.lam, gn, 0, alpha=0.6)
            ax.vlines(self.peaks[i]['cen'],0,self.peaks[i]['max'], linestyles='dashed', lw=0.5)
        ax.plot(self.lam, g, c='r', lw=2, label='Fit curve')
        ax.plot(self.lam, self.absorb, 'gray',alpha=0.6, label='Experimental data')
        ax.plot(self.lam, self.cache, 'b',alpha=0.6, label='Smoothed data')
        ax.legend(loc=0,fontsize=20)
        ax.set_xlabel('Wavelength (nm)',fontsize=20)
        ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        if title:
            ax.set_title(title, fontsize=20)
        plt.show()
            
    def no_exp_plot(self, title=''):
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        ax.plot(self.lam, self.g_total(self.lam), c='r', lw=2, label='Fit curve')
        #ax.plot(self.lam, self.absorb, 'gray',alpha=0.6, label='Experimental data')
        ax.plot(self.lam, self.cache, 'b',alpha=0.6, label='Smoothed data')
        for gn in self.g_group:
            ax.plot(self.lam, gn, alpha=0.6)
            ax.fill_between(self.lam, gn, 0, alpha=0.6)
        ax.legend(loc=0,fontsize=20)
        ax.set_xlabel('Wavelength (nm)',fontsize=20)
        ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
        ax.set_ylim(0,self.cache.max()*1.1)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        for p in self.peaks.values():
            if self.lam.min() <= p['cen'] <= self.lam.max():
                ax.vlines(p['cen'],0,p['max'], linestyles='dashed', lw=0.5)
        if title:
            ax.set_title(title, fontsize=20)
        plt.show()                 
        