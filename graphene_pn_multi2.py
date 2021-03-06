#!/usr/bin/env python
# coding: utf-8

from types import SimpleNamespace

# from ipywidgets import interact
import matplotlib
from matplotlib import pyplot
import numpy as np
import sys
import kwant
import time
import multiprocessing
import tinyarray
start = time.time()

W             = int(sys.argv[1])
H             = int(sys.argv[2])
maxE          = float(sys.argv[3])
mu_scattering = float(sys.argv[4])
mu_p          = float(sys.argv[5])
Delta         = float(sys.argv[6])
supra_ok      = float(sys.argv[7])
gamma         = float(sys.argv[8])

mu_n = mu_scattering
mu_S = 3      
t    = -3
phi  = np.pi

tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

graphene = kwant.lattice.general([[1, 0], [0, np.sqrt(3)]],  # lattice vectors
                                 [[0, 1/(2*np.sqrt(3))], [0, np.sqrt(3)/2], [1/2, 2/(np.sqrt(3))], 
                                  [1/2, 0]],norbs = 2)  # Coordinates of the sites
a,b,c,d = graphene.sublattices

rounded_height = round(H/np.sqrt(3))*np.sqrt(3)
lead_height = round(H/(2*np.sqrt(3)))*np.sqrt(3)

def square(pos):
    x, y = pos
    return x >= 0 and y >= 0              and x < W and y < rounded_height
def side_leads(pos):
    x, y = pos
    return x >= 0 and y >= lead_height    and x < W and y < rounded_height
def upper_row(pos):
    x, y = pos
    return x >= 0 and y >= rounded_height and x < W and y < rounded_height + np.sqrt(3)

def create_junction( t, on_site, mu_scattering, mu_n, mu_p, mu_S, Delta ):

    def pn_junction(site):
        x, y = site.pos
        if  (y > lead_height and y < rounded_height):
            return - mu_scattering  * tau_z
        else:
            return - (mu_p-(mu_p-mu_scattering)*y/lead_height) * tau_z
    
    sys = kwant.Builder(conservation_law=-tau_z, particle_hole = tau_y)
    sys[graphene.shape(square, (0,0))] =  pn_junction
    sys[graphene.neighbors()] = - t * tau_z
    
    if( supra_ok ):
        
        sys[graphene.shape(upper_row, (0,rounded_height))] = ( - mu_S ) * tau_z + np.exp(phi*1j) * Delta * tau_x
        top_row_index = round(H/np.sqrt(3))
        for i in np.arange(0,W):
            sys[d(i,top_row_index),a(i,top_row_index)]   = -t*tau_z
            sys[a(i,top_row_index),b(i,top_row_index)]   = -t*tau_z
            sys[b(i,top_row_index),c(i,top_row_index)]   = -t*tau_z
            if(i < W - 1):
                sys[d(i,top_row_index),a(i+1,top_row_index)]   = -t*tau_z
                sys[c(i,top_row_index),b(i+1,top_row_index)]   = -t*tau_z
        # gamma = - t
        # hopping between super and scatt. r.        
        for i in np.arange(0,W):
            sys[d(i,top_row_index),c(i,top_row_index-1)] = gamma*tau_z        

    #kwant.plot(sys);        

    negativeL = kwant.Builder(kwant.TranslationalSymmetry([1 ,0]), conservation_law=-tau_z, particle_hole = tau_y)
    negativeL[graphene.shape(side_leads, (0,H/2))] = ( - mu_n ) * tau_z
    negativeL[graphene.neighbors()] = - t * tau_z
    
    positiveL = kwant.Builder(kwant.TranslationalSymmetry([0 ,-np.sqrt(3)]), conservation_law=-tau_z, particle_hole = tau_y)
    positiveL[graphene.shape(square, (0,H/2))] = ( - mu_p ) * tau_z
    positiveL[graphene.neighbors()] = - t * tau_z
    
    superL = kwant.Builder(kwant.TranslationalSymmetry([0, +np.sqrt(3)]), conservation_law=-tau_z, particle_hole=tau_y)
    superL[graphene.shape(square, (0,0))] = ( - mu_S ) * tau_z + np.exp(phi*1j) * Delta * tau_x
    superL[graphene.neighbors()] = -t * tau_z    
        
    #kwant.plotter.bands(negativeL.finalized(),fig_size=(12,12),momenta=200,file="sideL_H"+str(H/2)+".png");
   
    sys.attach_lead(negativeL)
    sys.attach_lead(negativeL.reversed())
    sys.attach_lead(positiveL)
    if( supra_ok ):
        sys.attach_lead(superL)

    kwant.plot(sys,file="sys_W"+str(W)+"_H"+str(H)+"_supra"+str(int(supra_ok))+".png",fig_size=(10,10),dpi=100,site_color="black",show=True);  
        
    return sys

energy = np.linspace(-maxE, maxE, 1000)
junction = create_junction(t=t, on_site = 0, mu_scattering=mu_scattering, mu_n=mu_n, mu_p=mu_p, mu_S=mu_S, Delta=Delta )

T_11, T_11A, T_12, T_12A, T_13, T_13A = [],[],[],[],[],[]
for en in energy:
    smatrix  = kwant.smatrix(junction.finalized() , en, check_hermiticity = True)
    T_11.append(  smatrix.transmission((0,0), (0,0)))
    T_11A.append( smatrix.transmission((0,0), (0,1)))
    T_12.append(  smatrix.transmission((0,0), (2,0)))
    T_12A.append( smatrix.transmission((0,0), (2,1)))
    T_13.append(  smatrix.transmission((0,0), (1,0)))
    T_13A.append( smatrix.transmission((0,0), (1,1)))

pyplot.figure(figsize=(10,10))
#pyplot.subplot(2,2,1)
pyplot.plot(energy, T_12 ,"r-")
pyplot.plot(energy, T_12A,"b-")
pyplot.plot(energy, T_13 ,color="darkorange",linestyle="-")
pyplot.plot(energy, T_13A,"c-")
pyplot.plot([-mu_n, -mu_n],[0,max(T_13)],"b:");
pyplot.plot([-mu_p, -mu_p],[0,max(T_13)],"r:");
if( supra_ok ):
    pyplot.plot([ Delta, Delta],[0,max(T_13)],"k--");
    pyplot.plot([-Delta,-Delta],[0,max(T_13)],"k--");

pyplot.title("W="+str(W)+", H="+str(H)+", $\mu_{scatt.}$="+str(mu_scattering)+", $\mu_N$="+str(mu_n)+", $\mu_P$="+str(mu_p)+", $\Delta$="+str(Delta),fontsize=18);
pyplot.xlabel("$E\;[eV]$",fontsize=18)
pyplot.ylabel("$T$",fontsize=18)
pyplot.legend(["$T_{12}$","$T_{12A}$","$T_{13}$","$T_{13A}$","$\mu_N$","$\mu_P$","$\Delta$"],fontsize=18);
pyplot.xticks(fontsize=18); pyplot.yticks(fontsize=18);
pyplot.savefig("T_"+"_".join(sys.argv[1:])+".png");  

with open("./T_"+"_".join(sys.argv[1:])+".npy",'wb') as file:
    np.save(file,[energy,T_11,T_11A,T_12,T_12A,T_13,T_13A])

print("CORES:"+str(multiprocessing.cpu_count()))
print("RUNTIME:"+str(time.time()-start))
