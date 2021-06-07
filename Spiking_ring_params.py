#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:24:23 2020

@author: mikailkhona
"""

import numpy as np
import scipy as scipy


def hcat(A,B):
    return np.concatenate((A,B),axis=0)


def vcat(A,B):
    return np.concatenate((A,B),axis=1)


#Number of neurons_par
N_E = 2048
N_I = 512
Ntot = N_E + N_I

#pyramidal (E neurons)
Cm_E = 0.5 #nF/A
gL_E = 25 #nS/A
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

#concatenate these values for easier vectorization
v_thresh = hcat(v_th_E*np.ones(N_E) , v_th_I*np.ones(N_I))
Cm = hcat(Cm_E*np.ones(N_E) , Cm_I*np.ones(N_I))
v_reset = np.concatenate((v_res_E*np.ones(N_E) , v_res_I*np.ones(N_I)),axis=0)
tau_ref = np.array(np.concatenate((tau_ref_E*np.ones(N_E) , tau_ref_I*np.ones(N_I)),axis=0))
v_L = np.concatenate((E_L_E*np.ones(N_E),E_L_I*np.ones(N_I)),axis=0)
g_L = np.concatenate((gL_E*np.ones(N_E),gL_I*np.ones(N_I)),axis=0)
#g_L = hcat(E_L_E*np.ones(N_E) , E_L_I*np.ones(N_I))                                                               ]

#synaptic parameters (nS,ms)
#AMPA external excitatory
g_ext_E = 3.1
g_ext_I = 2.38

#NMDA (from E)
g_EE = 0.381
g_EI = 0.292
V_syn_NMDA = -70

#GABA (from I)
g_IE = 1.336
g_II = 1.024
V_syn_GABA = 0

taus_GABA = 10
taus_AMPA = 2
taus_NMDA = 100
taux_NMDA = 2

alphas = 0.5 #kHz for NMDA receptor

#synaptic weight matrix parameters(normalized) unitless
Jp_EE = 1.62
Jm_EE = 0.96
sigma_EE = (np.pi/180)*18 #degrees

#background input to neurons
v_ext = 1800/1000 #kHz

#simulation details
dt = 0.01 #ms
T = 3000 #ms
NT = round(T/dt)

#membrane voltages
v = -60. * np.ones((Ntot,1)).flatten()
#tracking previous spike
lastSpike = -100. * np.ones(Ntot)

#synaptic parameters

#N_ExN_E matrix with circulant ring attractor weights generated with kernel
#kernel function
def kernel(theta):
    return Jm_EE + (Jp_EE-Jm_EE)*np.exp(-((theta)**2)/(2*sigma_EE**2))


theta = np.linspace(0,2*np.pi,N_E)
theta = np.repeat(theta,N_E,0).reshape(N_E,N_E)
theta = theta - np.transpose(theta)
Ring_mask = kernel(np.minimum(np.mod(theta,2*np.pi),np.array(2*np.pi-np.mod(theta,2*np.pi))))
#pad ring_mask with zeros to match size of the whole network
Ring_mask = vcat(Ring_mask,np.ones((N_E,N_I)))
Ring_mask = np.concatenate((Ring_mask,np.ones((N_I,Ntot))),axis= 0)


#(i,j)th entry of mask = synaptic input from neuron i to j
#############################################@###############
#############################################@###############
#############################################@###############
#############################################@###############
#############################################@###############
######### Ring Attractor Mask ###############@### ones ######
#############################################@###############
#############################################@###############
############# N_E x N_E #####################@###############
#############################################@### E-->I #####
############### E-->E #######################@###############
#############################################@###############
#############################################@###############
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#############################################@###############
############# ones ##########################@###############
############ I-->E ##########################@## ones #######
#############################################@## I-->I ######
#############################################@###############

#gating variables

s_NMDA = np.zeros((N_E,1)).flatten();
x_NMDA = np.zeros((N_E,1)).flatten();
s_GABA = np.zeros((N_I,1)).flatten();
#reversal voltages
vsyn_NMDA = V_syn_NMDA*np.ones((N_E,1)).flatten();
vsyn_GABA = V_syn_GABA*np.ones((N_I,1)).flatten();
V_syn = np.concatenate((vsyn_NMDA , vsyn_GABA),axis = 0);

g_EE = g_EE*np.ones((N_E,N_E));
g_II = g_II*np.ones((N_I,N_I));
g_EI = g_EI*np.ones((N_E,N_I));
g_IE = g_IE*np.ones((N_I,N_E));

#conductance matrix: horizontally concatenate followed by vertical concatenation
g_E = np.concatenate((g_EE , g_EI),axis= 1);
g_I = np.concatenate((g_IE , g_II),axis = 1);
g =  np.concatenate((g_E,    g_I),axis = 0);

#excitatory inputs
s_AMPA = np.zeros((Ntot,1));
g_ext = np.concatenate((g_ext_E*np.ones((N_E,1)),g_ext_I*np.ones((N_I,1))),axis = 0)

#time to wait for transients to go away
trans = round(500/dt);

#external bump shaped stimulus to set bump location
I_stim = 130*vcat(np.ones((Ntot,trans//2)),np.zeros((Ntot,NT-trans//2))); #130pA
I_stim = np.einsum('ij,i->ij',I_stim,Ring_mask[:,1000])
