# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:58:26 2020



FOR UV-VIS SPECTRUM:
    STEP1: READ FILE:
        df = pd.read_csv(FILENAME, names=['x', 'y'], skiprows=2, sep='\t', )
        NOTICE: filename SHOULD INCLUDE THE PATH OF THE FILE
                lam, abs ARE ALIAS FOR WAVELENGTH AND ABSORBANCE, STRING
                df, ALIAS FOR RESTORED DATAFRAME
    STEP2: SELECT WINDOW:
        df = df[(df.x > BOTTOM) & (df.x < TOP)]
        NOTICE: bottom AND top ARE THE BOUNDARY OF THE WINDOW
    STEP3: DRAW PLOT:
        p, c, e = plot(df.x.values, df.y.values, N[, p0, title, label])
        NOTICE: p0, title, label ARE OPTIONAL ARGUMENTS, n IS THE NUMBER OF
                GAUSSIAN PACKET NEEDED, p,c,e STORES INFORMATION OF FITTED
                PARAMETERS, COVARIANCE AND ABSOLUTE ERROR
    STEP4: ANALYSIS:
        p_dict = all_peaks(p)
        compare(p_dict, INDEXA,INDEXB[,opt='norm'/'ratio'])
        NOTICE: p_dict IS A DICTIONARY, indexA AND indexB ARE THE KEY OF PEAKS
                YOU WANT TO COMPARE. opt IS THE METHOD YOU WANT TO CHOOSE.
        compare_plot(df.x.values, df.y.values, p_dict, (PEAK NUMBER))
        

@author: Xue FANG, Bo GAO
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, amp,cen,sigma):
    return amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen)/sigma)**2)))

def lorentzian(x, amp, cen, w):
    return amp/(1+(2*(x-cen)/w)**2)

def _n_gaussian(x, n, *args):
    """
    args should be [amp0, cen0, sigma0, amp1, cen1, sigma1, ..., ampn, cenn, sigman]
    """
    bias = args[0]
    args=args[1:]
    if len(args) != 3*n:
        raise ValueError('Not enough parameters!')
    g = 0
    for i in range(n):
        p = args[3*i:3*i+3]
        g += gaussian(x, *p)
    return g

def _n_lorentzian(x, n, *args):
    """
    args should be [amp0, cen0, sigma0, amp1, cen1, sigma1, ..., ampn, cenn, sigman]
    """
    if len(args) != 3*n:
        raise ValueError('Not enough parameters!')
    g = 0
    for i in range(n):
        p = args[3*i:3*i+3]
        g += lorentzian(x, *p)
    return g

def init_guess(n, bound=(400,650), p0=[]):
    """
    bound gives the range of wavelength
    p0 is a list of float or tuple with 3 elements(amp, center, sigma)
    """
    locs = np.linspace(bound[0], bound[1], n)
    amps = np.ones(n, dtype=float)
    sigmas = np.ones(n, dtype=float)
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
    return np.array([amps, locs, sigmas]).T.reshape(3*n)

def init_guess_lorentz(n, bound=(400,650), p0=[]):
    """
    bound gives the range of wavelength
    p0 is a list of float or tuple with 3 elements(amp, center, sigma)
    """
    locs = np.linspace(bound[0], bound[1], n)
    amps = np.ones(n, dtype=float)
    ws = np.ones(n, dtype=float)
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
                        ws[i] = p[2]
                        break 
            else:
                raise ValueError('Invalid input for p0!')
    return np.array([amps, locs, ws]).T.reshape(3*n)

def fit_n(xdata, ydata, n, p0=[], bias=0):
    bound = (xdata[0]+10,xdata[-1]-10)
    p_guess = np.insert(init_guess(n, bound=bound, p0=p0),0,bias)
    low_bounds = np.zeros(3*n+1)
    high_bounds = np.zeros(3*n+1)+np.inf
    for i in range(3*n+1):
        if i == 0:
            low_bounds[i]=min(0,np.min(ydata))
            high_bounds[i]=np.max(ydata)
        elif i%3 == 0:
            dx = (xdata[-1]-xdata[0])/n
            low_bounds[i]=1
            high_bounds[i]=0.66*dx
        elif i%3 == 2:
            low_bounds[i] = xdata[0]-50
            high_bounds[i] = xdata[-1]+50
    def g_func(x, *args):
        return _n_gaussian(x, n, *args)
    p_fit, cov = curve_fit(g_func, xdata, ydata, p0=p_guess, bounds=(low_bounds, high_bounds))
    err = np.mean((ydata-g_func(xdata, *p_fit))**2)
    return p_fit, cov, err

def fit_n_lorentz(xdata, ydata, n, p0=[]):
    bound = (xdata[0]+10,xdata[-1]-10)
    p_guess = init_guess_lorentz(n, bound=bound, p0=p0)
    def l_func(x, *args):
        return _n_lorentzian(x, n, *args)
    p_fit, cov = curve_fit(l_func, xdata, ydata, p0=p_guess, bounds=(0, np.inf))
    err = np.mean((ydata-l_func(xdata, *p_fit))**2)
    return p_fit, cov, err
    
def plot(xdata, ydata, n, p0=[], title='', label=''):
    p_fit, cov, err = fit_n(xdata, ydata, n, p0=p0)
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    g = _n_gaussian(xdata, n, *p_fit)
    p_fit_new=np.delete(p_fit,0)
    g_new = _n_gaussian(xdata,n, *np.insert(p_fit_new,0,0))
    p_group = p_fit_new.reshape(n,3)
    gs = [gaussian(xdata, *p) for p in p_group]
    
    if not label:
        label = 'Experimental data'
    ax.plot(xdata, ydata, c='k', lw=2, label=label)
    ax.plot(xdata, g, 'r--', label='Fit curve')
    ax.plot(xdata, g_new, 'b--', label='Fit no bias')
    for gn in gs:
        ax.plot(xdata, gn, alpha=0.6)
        ax.fill_between(xdata, gn, 0, alpha=0.6)
    ax.legend(loc=0,fontsize=20)
    ax.set_xlabel('Wavelength (nm)',fontsize=20)
    ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for p in p_group:
        if p[1]>=xdata[0] and p[1]<=xdata[-1]:
            ax.vlines(p[1],0,_n_gaussian(p[1],n,*p_fit), linestyles='dashed', lw=0.5)
    if title:
        ax.set_title(title, fontsize=20)
#    ax.text(600,np.max(ydata)/2,f'Error={err}')
    
    plt.show()
    print('bias:   ',p_fit[0],'error:   ',err)
    print(p_group)
    return p_fit, cov, err

def plot_lorentz(xdata, ydata, n, p0=[], title='', label=''):
    p_fit, cov, err = fit_n_lorentz(xdata, ydata, n, p0=p0)
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    g = _n_lorentzian(xdata, n, *p_fit)
    p_group = np.array(p_fit).reshape(n,3)
    gs = [lorentzian(xdata, *p) for p in p_group]
    
    if not label:
        label = 'Experimental data'
    ax.plot(xdata, ydata, c='k', lw=2, label=label)
    ax.plot(xdata, g, 'r--', label='Fit curve')
    for gn in gs:
        ax.plot(xdata, gn, alpha=0.6)
        ax.fill_between(xdata, gn, 0, alpha=0.6)
    ax.legend(loc=0,fontsize=20)
    ax.set_xlabel('Wavelength (nm)',fontsize=20)
    ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for p in p_group:
        if p[1]>=xdata[0] and p[1]<=xdata[-1]:
            ax.vlines(p[1],0,_n_lorentzian(p[1],n,*p_fit), linestyles='dashed', lw=0.5)
    if title:
        ax.set_title(title, fontsize=20)
#    ax.text(600,np.max(ydata)/2,f'Error={err}')
    
    plt.show()
    print(err)
    return p_fit, cov, err

def all_peaks(p_fit):
    n = len(p_fit)//3
    p_fit = p_fit.reshape(n,3)
    results = {}
    for i,p in enumerate(p_fit):
        loc = p[1]
        val = p[0]/(p[2]*np.sqrt(2*np.pi))
        results[i] = (loc, *p, val)
    return results

def compare(p_dict, index1, index2, opt='norm'):
    """
    norm option returns index1-index2/index1+index2
    ratio option retruns index1/index2
    """
    A = p_dict[index1][-1]
    B = p_dict[index2][-1]
    if opt.lower() == 'norm':
        return (A-B)/(A+B)
    elif opt.lower() == 'ratio':
        return A/B
    
def compare_plot(xdata, ydata, p_dict, indicies, label='', title=''):
    p_group=[]
    n = len(indicies)
    for index in indicies:
        p_group.append(p_dict[index][1:-1])
    p = np.array(p_group).reshape(3*n)
    g = _n_gaussian(xdata, n, *p)
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    if not label:
        label = 'Experimental data'
    ax.plot(xdata, ydata, 'k', lw=2, label=label)
    ax.plot(xdata, g, 'r--', label='Fit curve')
    
    gs = [gaussian(xdata, *_p) for _p in p_group]
    for gn in gs:
        ax.plot(xdata, gn, alpha=0.6)
        ax.fill_between(xdata, gn, 0, alpha=0.6)
    ax.legend(loc=1,fontsize=20)
    ax.set_xlabel('Wavelength (nm)',fontsize=20)
    ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for _p in p_group:
        if _p[1]>=xdata[0] and _p[1]<=xdata[-1]:
            ax.vlines(_p[1],0,_n_gaussian(_p[1],n,*p), linestyles='dashed', lw=0.5)
    if title:
        ax.set_title(title, fontsize=20)
    plt.show()
        
def compare_plot_lorentz(xdata, ydata, p_dict, indicies, label='', title=''):
    p_group=[]
    n = len(indicies)
    for index in indicies:
        p_group.append(p_dict[index][1:-1])
    p = np.array(p_group).reshape(3*n)
    g = _n_lorentzian(xdata, n, *p)
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    if not label:
        label = 'Experimental data'
    ax.plot(xdata, ydata, 'k', lw=2, label=label)
    ax.plot(xdata, g, 'r--', label='Fit curve')
    
    gs = [lorentzian(xdata, *_p) for _p in p_group]
    for gn in gs:
        ax.plot(xdata, gn, alpha=0.6)
        ax.fill_between(xdata, gn, 0, alpha=0.6)
    ax.legend(loc=1,fontsize=20)
    ax.set_xlabel('Wavelength (nm)',fontsize=20)
    ax.set_ylabel('Absorbance (a.u.)',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for _p in p_group:
        if _p[1]>=xdata[0] and _p[1]<=xdata[-1]:
            ax.vlines(_p[1],0,_n_lorentzian(_p[1],n,*p), linestyles='dashed', lw=0.5)
    if title:
        ax.set_title(title, fontsize=20)
    plt.show()

