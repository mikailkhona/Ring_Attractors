#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import time as time
from scipy.special import erf
from numpy.fft import rfft, irfft
import math as math
from collections import deque

def hcat(A,B):
    return np.concatenate((A,B),axis=0)


def vcat(A,B):
    return np.concatenate((A,B),axis=1)

def NMDA_modulation(V):
    return (1/(1+np.exp(-0.062*V)/3.57))

def gaussian(x,mean,sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))

def convolve_sNMDA(fft_presyn_weight_kernel,s_NMDA):
    g_EE_s_NMDA = irfft(fft_presyn_weight_kernel*rfft(s_NMDA))
    return g_EE_s_NMDA


def RK2(v,timestep,dynamics_coeff,delta_function_input):
    v = v*(1+timestep*dynamics_coeff +0.5*(timestep**2)*(dynamics_coeff**2)) + delta_function_input
    return v

def RK2_NMDA(s_NMDA,timestep,C_coeff,D_vec):
    s_NMDA_temp  = s_NMDA + 0.5*timestep*(C_coeff*s_NMDA + D_vec*(1-s_NMDA))
    s_NMDA = s_NMDA +timestep*(C_coeff*s_NMDA_temp + D_vec*(1-s_NMDA_temp))
    return s_NMDA


N_E = 1024
N_I = 512
Ntot = N_E + N_I
f = 2048//N_E #scale factor for conductances
#pyramidal (E neurons)
Cm_E = 0.5 #nF
gL_E = 25 #nS
E_L_E = -70 #mv
v_th_E = -50 #mv
v_res_E = -60 #mv
tau_ref_E = 2 #ms

#interneurons (I neurons)
Cm_I = 0.2
gL_I = 20
E_L_I = -70
v_th_I = -50
v_res_I = -60
tau_ref_I = 1
                                                       

#synaptic parameters (nS,ms)
#NMDA (from E)
g_EE = 0.381*f  #nS
g_EI = 0.292*f  #nS
V_syn_NMDA = 0

#GABA (from I)
g_IE = 1.336*f  #nS
g_II =  1.024*f #nS
V_syn_GABA = -70

#AMPA
#AMPA external excitatory
g_ext_E = 3.1  #nS
g_ext_I = 2.38 #nS
#AMPA recurrent excitatory
g_AMPA_E = 0.104*f
g_AMPA_I = 0.081*f

#parameters from renart

'''
#NMDA (from E)
g_EE = 931/N_E  #nS
g_EI = 750/N_E  #nS
V_syn_NMDA = 0

#GABA (from I)
g_IE =1024/N_I  #nS
g_II = 835/N_I  #nS
V_syn_GABA = -70
'''
#parameters from compte
taus_GABA = 10
taus_AMPA = 2  #ms
taus_NMDA = 100
taux_NMDA = 2

alphas = 0.5 #kHz for NMDA receptor

#background input to neurons
v_ext = 1800/1000 #kHz

#simulation details
dt = 0.01 #ms
T = 3000 #ms
NT = round(T/dt)

#membrane voltages
v = -51. * np.ones((Ntot))
v_E = v[0:N_E]
v_I = v[N_E:Ntot]
#tracking previous spike
lastSpike_E = -tau_ref_E*np.ones((N_E))
lastSpike_I = -tau_ref_I*np.ones((N_I))

#synaptic parameters

#N_ExN_E matrix with circulant ring attractor weights generated with kernel
#synaptic weight matrix parameters(normalized) unitless
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
# validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
weight_profile_45 = deque(presyn_weight_kernel)
rot_dist = int(round(len(weight_profile_45) / 8))
weight_profile_45.rotate(rot_dist)

time_axis = np.linspace(0,NT*dt,NT)

#(i,j)th entry of mask = synaptic input from neuron j to i
#############################################@###############
#############################################@###############
#############################################@###############
#############################################@###############
#############################################@###############
######### Ring Attractor Mask ###############@### ones ######
#############################################@###############
#############################################@###############
############# N_E x N_E #####################@###############
#############################################@### I-->E #####
############### E-->E #######################@###############
#############################################@###############
#############################################@###############
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#############################################@###############
############# ones ##########################@###############
############ E-->I ##########################@## ones #######
#############################################@## I-->I ######
#############################################@###############

#gating variables

s_NMDA = 0.05*np.ones((N_E))
x_NMDA = 0.1*np.ones((N_E))
s_GABA = 0.5*np.ones((N_I))
s_AMPA_rec = 0.05*np.ones((N_E))


#excitatory inputs
s_AMPA_ext_E = 0.6*np.ones((N_E));
s_AMPA_ext_I = 0.6*np.ones((N_I));


#time to wait for transients to go away
trans = round(500/dt);

#external bump shaped stimulus to set bump location
I_stim_E = 450*gaussian(np.linspace(0,Ntot,Ntot),0.4*Ntot,0.05*Ntot)[0:N_E]

#storing spike times for all neurons
times_E = np.zeros((N_E,NT))
times_I = np.zeros((N_I,NT))
t = 0

binsize = round(100/dt)  # no of milliseconds to bin spikes with
firing_rates = np.zeros((N_E,NT-binsize))

#prefactor for firing rate after binning ( = 1/total time binned in seconds to get Hz)
pre = 1000/(dt*binsize)
#storing synaptic voltage of center pyramidal neuron
peak= np.zeros((NT))

#start simulation
for tt in range(0,NT):
    
    prevV_E = v_E.copy()
    prevV_I = v_I.copy()
    
    t = dt*tt #in ms

    #check % complete
    if np.mod(tt,NT/100) == 1: #print percent complete every 1%
        print("\r",round(100*tt/NT),"%")
        #plt.plot(firing_rates[:,tt])
        #plt.title(str(t)+'ms')
        #plt.show()
        
        
    #E neurons
    #update synaptic parameters (decay)
    s_NMDA =  RK2_NMDA(s_NMDA,dt,-1/taus_NMDA,alphas*x_NMDA)
    x_NMDA =  RK2(x_NMDA,dt,-1/taux_NMDA,0)
    s_AMPA_ext_E =  RK2(s_AMPA_ext_E,dt,-1/taus_AMPA,0)


    #calculate synaptic current to each neuron from all sources
    #by summing over rows,
    I_syn_E_NMDA = g_EE*(v_E-V_syn_NMDA)*convolve_sNMDA(fft_presyn_weight_kernel,s_NMDA)*NMDA_modulation(v_E)
    I_syn_E_GABA = g_IE*(v_E - V_syn_GABA)*np.sum(s_GABA)
    
    #do input spikes
    #bools = np.random.rand((N_E,1))<dt*v_ext
    s_AMPA_ext_E += (np.random.uniform(0,1,(N_E,1))).flatten()<dt*v_ext#np.random.poisson(dt*v_ext,(N_E,1)).flatten()
    
    #external input from spikes  = g_AMPA*s_AMPA*(V_post-V_rev_AMPA)
    I_ext_E = g_ext_E*s_AMPA_ext_E*v_E
    
    #check refractory period and update voltage using total current to each neuron
    I_E = (t>(tau_ref_E + lastSpike_E))*(-gL_E*(v_E-E_L_E) - I_syn_E_NMDA - I_syn_E_GABA - I_ext_E + I_stim_E)
    v_E = v_E + 0.001*dt*I_E*(Cm_E)**(-1)
    
    #check spike
    spikeBool_E = (v_E>v_th_E)

    #update spike times
    times_E[:,tt] = spikeBool_E

    #reset voltage - multiply by 0 if spike (1-bool) and add v_reset (bool*v_reset)
    v_E[spikeBool_E] = v_res_E
    lastSpike_E[spikeBool_E] = t
    
    #store membrane voltage of peak neuron
    peak[tt] = v_E[int(0.4*Ntot)]

    #do spike (increment gating variables by 1 if spike occured at that synapse)
    x_NMDA += spikeBool_E
    
    
    #I neurons
    #update synaptic parameters (decay)
    s_GABA =  RK2(s_GABA,0.001*dt,-s_GABA/taus_GABA,0)
    s_AMPA_ext_I =  RK2(s_AMPA_ext_I,dt,-s_AMPA_ext_I/taus_AMPA,0)
    
    #calculate synaptic current to each neuron from all sources
    #by summing over rows,
    I_syn_I_NMDA = g_EI*(v_I-V_syn_NMDA)*NMDA_modulation(v_I)*np.sum(s_NMDA)
    I_syn_I_GABA = g_II*(v_I - V_syn_GABA)*np.sum(s_GABA)
    
    #do input spikes
    s_AMPA_ext_I += (np.random.uniform(0,1,(N_I,1))).flatten()<dt*v_ext#np.random.poisson(dt*v_ext,(N_I,1)).flatten()
    
    #external input from spikes  = g_AMPA*s_AMPA*(V_post-V_rev_AMPA)
    I_ext_I = g_ext_I*s_AMPA_ext_I*v_I
    
    #check refractory period and update voltage using total current to each neuron
    I_I = (t>(tau_ref_I + lastSpike_I))*(-gL_I*(v_I-E_L_I) - I_syn_I_NMDA - I_syn_I_GABA - I_ext_I)
    v_I = v_I + 0.001*dt*I_I*(Cm_I)**(-1)
    
    #check spike
    spikeBool_I = (v_I>v_th_I)

    #update spike times
    times_I[:,tt] = spikeBool_I

    #reset voltage - multiply by 0 if spike (1-bool) and add v_reset (bool*v_reset
    v_I[spikeBool_I] = v_res_I
    lastSpike_I[spikeBool_I] = t 

    #update synaptic variables following spike
    s_GABA += spikeBool_I
    
    # correction to membrane voltage after spike from Hansel et al 1998
    # in time [t,t+dt] if a spike occurs, there's an error of dt in spike time, 
    # this scheme linearly interpolates between t and t+dt for t_spike
    v_E[spikeBool_E] = (v_E[spikeBool_E]-v_th_E)*(1+gL_E*(prevV_E[spikeBool_E]-v_res_E)/(v_E[spikeBool_E]-prevV_E[spikeBool_E])) +v_res_E
    v_I[spikeBool_I] = (v_I[spikeBool_I]-v_th_I)*(1+gL_I*(prevV_I[spikeBool_I]-v_res_I)/(v_I[spikeBool_I]-prevV_I[spikeBool_I])) +v_res_I

    if(tt>binsize):
        firing_rates[:,tt] = pre*np.sum(times_E[:,tt-binsize:tt],axis=1)
        
    
    




