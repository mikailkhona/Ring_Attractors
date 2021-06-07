#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:45:36 2020

@author: mikailkhona
"""
import numpy as np
from scipy.special import erf
from numpy.fft import rfft, irfft
from collections import deque
import matplotlib.pyplot as plt

k1 = 1
k2 = 0.3
tau = 30

#Weight Matrix function
def w(x,a):
    return a * np.exp(k1 * (np.cos(x) - 1)) - a * np.exp(k2 * (np.cos(x) - 1))

def phi(a):
    return 0.04*np.exp(a)/tau
def sigm(a):
    sigm = 1. / (1. + np.exp(-a))
    return 0.040*sigm

def relu(a):
    return (a+np.abs(a))/(2*tau)

def gaussian(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))

def RK2(v,timestep,dynamics_coeff,delta_function_input):
    v = v*(1+timestep*dynamics_coeff +0.5*(timestep**2)*(dynamics_coeff**2)) + delta_function_input
    return v



N_E = 1024
N_I = 256
N = N_E + N_I

Jp_EE = 1.62
sigma_EE = 14.4 #degrees
#kernel function
tmp = np.sqrt(2. * np.pi) * sigma_EE * erf(180. / np.sqrt(2.) / sigma_EE) / 360.
Jm_EE = (1. - Jp_EE * tmp) / (1. - tmp)
presyn_weight_kernel = \
    [(Jm_EE +
      (Jp_EE - Jm_EE) *
      np.exp(-.5 * (360. * min(j, N_E - j) / N_E) ** 2 / sigma_EE** 2))
     for j in range(N_E)]
    
presyn_weight_kernel = np.array(presyn_weight_kernel)    
# validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
fft_presyn_weight_kernel = rfft(presyn_weight_kernel)

W_EE = 1./N
W_EI = 1./N
W_IE = 1.1/N
W_II = 1.1/N

tau_E = tau
tau_I = tau
omg1 = 2 * np.pi / 100      # theta oscillation  10Hz = 2 * pi/100
omg2 = omg1    #inputs and external inputs oscillation at the same frequency
dt = 0.1 #      time step = 0.1msec

# total time of simulation
t_stop = 3000
I_stim = np.roll(presyn_weight_kernel,N_E//2)
NT = int(t_stop/dt)

s_E = np.random.uniform(-1,1,N_E)
s_I = 0.1*np.ones((N_I))

bg = 3
b = bg*np.ones(NT)#*(1+np.cos(omg1*np.linspace(0,t_stop,NT)))

S = np.zeros((N,NT))
total_inhib = np.zeros(NT)

for tt in range(0,NT):
    t = dt*tt
    if np.mod(tt,NT/10) == 1: #print percent complete
        print("\r",round(100*tt/NT),"%")
        plt.plot(s_E)
        plt.show()
    
    #transient current lasts only 800ms
    if(t>800):
        I_stim = 0
        
    
    s_E = s_E + dt*(-s_E/tau_E + phi(W_EE*irfft(rfft(s_E)*fft_presyn_weight_kernel) - W_IE*np.sum(s_I) + I_stim))
    s_I = s_I + dt*(-s_I/tau_I + phi(-W_II*np.sum(s_I)+ W_EI*np.sum(s_E)+b[tt]))
    S[0:N_E,tt] = s_E
    S[N_E:,tt] = s_I
    total_inhib[tt] = np.sum(s_I)
   
plt.subplot(121)    
plt.plot(total_inhib)
plt.subplot(122)
plt.plot(S[N//2,:])
    
    
    
    
    
    
    
    
    
    
    
    
    