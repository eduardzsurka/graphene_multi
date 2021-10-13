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
s_f           = int(sys.argv[3])
maxE          = float(sys.argv[4])
mu_scattering = float(sys.argv[5])
mu_p          = float(sys.argv[6])
Delta         = float(sys.argv[7])
supra_ok      = float(sys.argv[8])

# trigonal wrapping starts to become substantial at ~ 800 meV,
# for us this means (due to the scaling of the hopping) we shouldn't
# exceed 800/20 = 40 meV in energies
mu_n = mu_scattering
mu_S = 0.04          # 40 meV
t    = -3/s_f        # scaling accoring to Phys. Rev. Lett. 114, 036601 (2015)
phi  = np.pi

tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

Delta_mat = tinyarray.array([[0, Delta*np.exp(phi*1j)], [Delta*np.exp(-phi*1j) , 0]])

graphene = kwant.lattice.general([[1, 0], [0, np.sqrt(3)]],  # lattice vectors
                                 [[0, 1/(2*np.sqrt(3))], [0, np.sqrt(3)/2], [1/2, 2/(np.sqrt(3))], 
                                  [1/2, 0]],norbs = 2)  # Coordinates of the sites

rounded_height = round(H/np.sqrt(3))*np.sqrt(3)
lead_height = round(H/(2*np.sqrt(3)))*np.sqrt(3)

def square(pos):
    x, y = pos
    return x >= 0 and y >= 0              and x < W and y < rounded_height
def side_leads(pos):
    x, y = pos
    return x >= 0 and y >= lead_height    and x < W and y < rounded_height

def create_junction( t, on_site, mu_scattering, mu_n, mu_p, mu_S, Delta ):

    def pn_junction(site):
        x, y = site.pos
        if  (y > lead_height ):
            return - mu_scattering  * tau_z
        else:
            return - (mu_p-(mu_p-mu_scattering)*y/lead_height) * tau_z
    
    sys = kwant.Builder(conservation_law=-tau_z, particle_hole = tau_y)
    sys[graphene.shape(square, (0,0))] =  pn_junction
    sys[graphene.neighbors()] = - t * tau_z

    #kwant.plot(sys);        

    negativeL = kwant.Builder(kwant.TranslationalSymmetry([1 ,0]), conservation_law=-tau_z, particle_hole = tau_y)
    negativeL[graphene.shape(side_leads, (0,H/2))] = ( - mu_n ) * tau_z
    negativeL[graphene.neighbors()] = - t * tau_z
    
    positiveL = kwant.Builder(kwant.TranslationalSymmetry([0 ,-np.sqrt(3)]), conservation_law=-tau_z, particle_hole = tau_y)
    positiveL[graphene.shape(square, (0,H/2))] = ( - mu_p ) * tau_z
    positiveL[graphene.neighbors()] = - t * tau_z
    
    superL = kwant.Builder(kwant.TranslationalSymmetry([0, +np.sqrt(3)]))
    superL[graphene.shape(square, (0,0))] = ( - mu_S ) * tau_z + Delta_mat
    superL[graphene.neighbors()] = - t * tau_z    
        
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
Junction = junction.finalized()

energies = np.append(np.linspace(-Delta,Delta,41),0.0003283283283283273)
J_0 = kwant.operator.Current(Junction, tau_z) #hole part is taken into account with a - sign

for en in energies:
    scattering_states = kwant.solvers.default.wave_function(Junction,energy=en)

    Psi = scattering_states(1)  # scattering state from lead 1
    i = 0
    psis = []

    while True:
        try:
           psis.append(Psi[i])
           i=i+1
        except:
            break
    n = i
    fig, ax = pyplot.subplots(nrows=2,ncols=n)
    fig.set_figheight(7*2)
    fig.set_figwidth(3.5*(n+1))
    fig.set_tight_layout("pad")

    fig_file = "wf_"+"_".join(sys.argv[1:])+"_E"+"{:.6f}".format(en)+".png"
    for i in np.arange(0,n):
        psi = psis[i].reshape(-1,2)

        density0 = (abs(psi[:,0])**2)# electron part
        density1 = (abs(psi[:,1])**2)# hole part
        #norm = sum(density0) + sum(density1)
        norm = 1
        ax[0,i].set_title("Mode "+str(i)+", e. part, $|\psi|^2=$"+str(round(sum(density0/norm),4)))
        kwant.plotter.density(Junction, density0/norm, vmin=0, vmax=1, relwidth=0.02, fig_size=(4,7),ax=ax[0,i],cmap="inferno")

        ax[1,i].set_title("Mode "+str(i)+", h. part, $|\psi|^2=$"+str(round(sum(density1/norm),4)))
        kwant.plotter.density(Junction, density1/norm, vmin=0, vmax=1, relwidth=0.02, fig_size=(4,7),ax=ax[1,i],cmap="inferno")
    fig.savefig(fname=fig_file)

#        current = J_0(psis[i])
#        c.append(current)
#        kwant.plotter.current(Junction,current,fig_size=(7,10),file="current_"+"_".join(sys.argv[1:])+"_mode"+str(i)+"_E"+"{:.6f}".format(en)+".png")
#        kwant.plotter.current(Junction,c[0]+c[1],fig_size=(7,10),file="current_"+"_".join(sys.argv[1:])+"_E"+"{:.6f}".format(en)+".png")


print("CORES:"+str(multiprocessing.cpu_count()))
print("RUNTIME:"+str(time.time()-start))
