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

h = 4.136e-15 #ev*s
v_F = 1e6 #m/S
lattice_c = 2.46*1e-10# lattice constant in m

s_f  = 20
dist = 100 #nm
dist_nm = dist/(lattice_c*1e9*s_f)

W_S           = int(sys.argv[1])
H_L           = round(int(sys.argv[2])/np.sqrt(3))*np.sqrt(3)
H             = round(int(sys.argv[3])/np.sqrt(3))*np.sqrt(3)
H_pn          = round(int(sys.argv[4])/np.sqrt(3))*np.sqrt(3)
maxE          = float(sys.argv[5])
mu_scattering = float(sys.argv[6])
mu_p          = float(sys.argv[7])
Delta         = float(sys.argv[8])
supra_ok      = int(sys.argv[9])

#print("H_L="+str(H_L))
#print("H="+str(H))

# trigonal wrapping starts to become substantial at ~ 800 meV,
# for us this means (due to the scaling of the hopping) we shouldn't
# exceed 800/20 = 40 meV in energies

mu_n = mu_scattering
mu_S = 0.04          # 40 meV
t    = -3/s_f        # scaling accoring to Phys. Rev. Lett. 114, 036601 (2015)
phi  = np.pi

#evaluate Fermi wavelength and size of pn junction

lambda_F = h*v_F/mu_n
H_pn_m = H_pn*lattice_c* s_f
d_pn   = H_pn_m/lambda_F

#evaluate Thouless energy (from Phys. Rev. B 97, 045421 (2018), page 9, 1st paragraph)

#E_Th = h*v_F/(2*np.pi * (2*(H+H_L)*lattice_c* s_f) )

#evaluate coherence length

xi_0 = h*v_F/(Delta*2*np.pi)

tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

Delta_mat = tinyarray.array([[0, Delta*np.exp(phi*1j)], [Delta*np.exp(-phi*1j) , 0]],complex)

graphene = kwant.lattice.general([[1, 0], [0, np.sqrt(3)]],  # lattice vectors
                                 [[0, 1/(2*np.sqrt(3))], [0, np.sqrt(3)/2], [1/2, 2/(np.sqrt(3))], 
                                  [1/2, 0]],norbs = 2)  # Coordinates of the sites
def Color(site):
    y = site.pos[1]/H_pn
    if(y<1):
        return [1-y,0,y]
    else:
        return [0,0,1]

def square(pos):
    x, y = pos
    return x >= 0 and x < W_S and y >= 0 and y < H + H_L - 1e-6
def side_leads(pos):
    x, y = pos
    return x >= 0 and x < 1   and y >= H and y < H + H_L - 1e-6# + 0.5

def create_junction( t, on_site, mu_scattering, mu_n, mu_p, mu_S, Delta, param_string ):

    def pn_junction(site):
        x, y = site.pos
        if  (y > H_pn ):
            return - mu_scattering  * tau_z
        else:
            return - ( mu_p - (mu_p-mu_scattering)*y/H_pn ) * tau_z

    sys = kwant.Builder(conservation_law=-tau_z, particle_hole = tau_y)
    sys[graphene.shape(square, (0,0))] =  pn_junction
    sys[graphene.neighbors()] = - t * tau_z

    #kwant.plot(sys);

    negativeL = kwant.Builder(kwant.TranslationalSymmetry([1 ,0]), conservation_law=-tau_z, particle_hole = tau_y)
    negativeL[graphene.shape(side_leads, (0,H))] = ( - mu_n ) * tau_z
    negativeL[graphene.neighbors()] = - t * tau_z

    positiveL = kwant.Builder(kwant.TranslationalSymmetry([0 ,-np.sqrt(3)]), conservation_law=-tau_z, particle_hole = tau_y)
    positiveL[graphene.shape(square, (0,H))] = ( - mu_p ) * tau_z
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

    # SZÍNSKÁLÁT!
    kwant.plot(sys,file="sys_"+param_string+".png",dpi=100,site_color=Color,show=True,ax = ax,site_size=0.4,lead_color="black");

    gca = fig.gca()
    gca.set_aspect("equal")

    ax.plot([-2.5,-2.5+dist_nm],[-2.5,-2.5],lw=3,color="black");
    ax.text(-2.5,-1,str(dist)+" nm")
    fig.savefig(fname="sys_"+param_string+".png",)

    return sys

fig, ax = pyplot.subplots(nrows=1,ncols=1)
aspect = (20+W_S)/(20+H+H_L)
dim = 16
if(aspect>1):
    sys_fig_height = dim/aspect
    sys_fig_width  = dim
else:
    sys_fig_height = dim
    sys_fig_width  = dim*aspect
    
fig.set_tight_layout("pad")
fig.set_figheight( sys_fig_height )
fig.set_figwidth(  sys_fig_width  )
    
fig.set_tight_layout("pad")
fig.set_figheight( sys_fig_height )
fig.set_figwidth(  sys_fig_width  )


energy = np.linspace(-maxE, maxE, 1000)
junction = create_junction(t=t, on_site = 0, mu_scattering=mu_scattering, mu_n=mu_n, mu_p=mu_p, mu_S=mu_S, Delta=Delta, param_string= "_".join(sys.argv[1:5])+"_"+str(supra_ok))

#exit()

T_11, T_11A, T_12, T_12A, T_13, T_13A = [],[],[],[],[],[]
for en in energy:
    smatrix  = kwant.smatrix(junction.finalized() , en, check_hermiticity = True)
    T_11.append(  smatrix.transmission((0,0), (0,0)))
    T_11A.append( smatrix.transmission((0,0), (0,1)))
    T_12.append(  smatrix.transmission((0,0), (2,0)))
    T_12A.append( smatrix.transmission((0,0), (2,1)))
    T_13.append(  smatrix.transmission((0,0), (1,0)))
    T_13A.append( smatrix.transmission((0,0), (1,1)))

Max = max([max(T_11),max(T_11A),max(T_12),max(T_12A),max(T_13),max(T_13A)])
Max_ys = [-Max*0.05,Max*1.05]

pyplot.figure(figsize=(10,10))
#pyplot.subplot(2,2,1)
pyplot.plot(energy/Delta, T_12 ,"r-")
pyplot.plot(energy/Delta, T_12A,"b-")
pyplot.plot(energy/Delta, T_13 ,color="darkorange",linestyle="-")
pyplot.plot(energy/Delta, T_13A,"c-")
pyplot.plot([-mu_p/Delta, -mu_p/Delta],Max_ys,"r:");
pyplot.plot([-mu_n/Delta, -mu_n/Delta],Max_ys,"b:");
pyplot.xlim([-maxE/Delta,maxE/Delta]);
if( supra_ok ):
    pyplot.plot([ 1, 1],Max_ys,"k--");
    pyplot.plot([-1,-1],Max_ys,"k--");

pyplot.title("$W_S="+str(W_S)+",\, H_L="+str(round(H_L,1))+",\, H="+str(round(H,1))+",\, H_{pn}="+str(H_pn)+
             ",$\n$\mu_{scatt.}=\mu_N="+str(int(mu_n/Delta))+"$ meV$,\, \mu_P="+str(int(mu_p/Delta))+"$ meV$,\, \Delta="+str(int(Delta/Delta))+"$ meV$"+
             ",$\n$H_{pn}/\lambda_F^N="+str(int(H_pn_m*1e9))+"/"+str(int(lambda_F*1e9))+"="+"{:.3f}".format(d_pn)+
#              ",$\n$H_{pn}/\lambda_F="+"{:.1f}".format(d_pn)+
             ",\,\Delta/E_{Th}\propto (H+H_L)/\\xi_0="+"{:.3f}".format( (H+H_L)*lattice_c*s_f / (xi_0) )+ #Delta/E_Th)+
             "$",fontsize=18);

pyplot.xlabel("$E\;$[meV]",fontsize=18)
pyplot.ylabel("$T$",fontsize=18)
pyplot.legend(["$T_{12}$","$T_{12A}$","$T_{13}$","$T_{13A}$","$\mu_P$","$\mu_N$","$\Delta$"],fontsize=18);
pyplot.xticks(fontsize=18); pyplot.yticks(fontsize=18);
pyplot.savefig("T_"+"_".join(sys.argv[1:])+".png");

with open("./T_"+"_".join(sys.argv[1:])+".npy",'wb') as file:
    np.save(file,[energy,T_11,T_11A,T_12,T_12A,T_13,T_13A])

print("CORES:"+str(multiprocessing.cpu_count()))
print("RUNTIME:"+str(time.time()-start))
