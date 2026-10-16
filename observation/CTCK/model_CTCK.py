# %%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["NUMBA_THREADING_LAYER"] = "tbb"

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import gc

import cmocean
from netCDF4 import Dataset
#from datetime import date, timedelta
import xarray as xr
## . . For animation
from matplotlib import animation
from IPython.display import HTML, Image
import geopandas
import rioxarray as rio
from shapely.geometry import mapping, shape
import pandas as pd
from pyproj import Transformer
from numba import jit, prange, set_num_threads, threading_layer, get_num_threads
n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
set_num_threads(n_threads)
import time

# %% [markdown]
# # Load Data

# %%
path = "/home/m10936923/scratch"

# %%
IMBIE = geopandas.read_file(path+"/Inputs/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp", crs="epsg:3031")
scripps = geopandas.read_file(path+"/Inputs/scripps_GL/scripps_antarctica_polygons_v1.shp", crs="epsg:3031")
scripps_gl = scripps.loc[scripps['Id'].astype(int) == 1]

# Open each file, which are quadrants in polar stereographic coordinations around the Geographic South Pole
qs = ['A1','A2','A3','A4']
ds_smb = {q: xr.open_dataset(f'{path}/Inputs/ICESat2/gsfc_ATL15_{q}_0329_10km_005_02.nc') for q in qs}
SMB = (ds_smb['A1']
         .combine_first(ds_smb['A2'])
         .combine_first(ds_smb['A3'])
         .combine_first(ds_smb['A4']))

nt,ny,nx = np.shape(SMB.zsurf)
nt = 22
smb_lag1 = np.zeros((nt,ny,nx),dtype=float)
for i in range(0,nt):
    #difference of adjacent surfaces divided by 0.25, a quarter of the year
    dsmb_dt = (SMB.zsurf[i+1,:,:] - SMB.zsurf[i,:,:])/0.25
    smb_lag1[i,:,:] = dsmb_dt

ds_atl = {q: xr.open_dataset(f'{path}/Inputs/ICESat2/ATL15_{q}_0329_10km_005_02.nc',
                         group='dhdt_lag1') for q in qs}

ATL15 = (ds_atl['A1']
         .combine_first(ds_atl['A2'])
         .combine_first(ds_atl['A3'])
         .combine_first(ds_atl['A4']))


for i in range(0,nt):
    ATL15.dhdt[i,:,:] = ATL15.dhdt[i,:,:] - smb_lag1[i,:,:]

ds_dh = {q: xr.open_dataset(f'{path}/Inputs/ICESat2/ATL15_{q}_0329_10km_005_02.nc',
                         group='delta_h') for q in qs}

dh = (ds_dh['A1']
         .combine_first(ds_dh['A2'])
         .combine_first(ds_dh['A3'])
         .combine_first(ds_dh['A4']))

for i in range(0,nt):
    dh.delta_h[i,:,:] = dh.delta_h[i,:,:] - SMB.zsurf[i,:,:]

# %%
bedmachine = xr.open_dataset(path+"/Inputs/BedMachineAntarctica-v3.nc")
bm_x, bm_y = np.meshgrid(bedmachine.x, bedmachine.y)
atl15_x, atl15_y = np.meshgrid(ATL15.x, ATL15.y)
interp = RegularGridInterpolator(points=[bedmachine.x.values,bedmachine.y.values], values=bedmachine["thickness"].values.T, method="linear", fill_value=np.nan, bounds_error = False)
bm_new = interp((atl15_x, atl15_y))

# %%
h_init_xr = xr.open_dataarray(path+"/Outputs/observation/spinup_head_CTCK.nc")
h_init = h_init_xr.values

# %% [markdown]
# # Parameters

# %%
nt,ny,nx = np.shape(ATL15["dhdt"])

year = 3600*24*365.25

xi = 0.2 #Skempton's constant
rhoi = 920
rhow = 1000
g = 9.81
mu_w = 1e-3
S = 1e-6
perm=1e-13
K_t = perm*rhow*g/mu_w
D = K_t/S

L = 400e3
dz = 100
z = np.arange(0,L+dz,dz)
nz = np.size(z)
dt = 3600*2
tf = year*5.5
m_nt = int(np.round(tf/dt) + 1)
t = np.linspace(0,tf,num=m_nt-1)

#save each 5 days (10 timesteps)
intSave = 60

# %% [markdown]
# # Forcing

# %%
model_dates = np.arange(dh.time.values[0], dh.time.values[nt-1], np.timedelta64(2,'h'))
icesat2_dates = ATL15.time[0:nt]

dhdt_forcing = ATL15["dhdt"][0:nt,:,:].values

ncols = nx*ny

dhdt_forcing_int = np.zeros((len(model_dates),ny,nx))
thk_forcing_int = np.zeros((len(model_dates), ny, nx))
thk_forcing_int[0, :, :] = bm_new
interp_dt = dt/year #dt of 0.5 days in year

for col in range(ncols):
        x_idx = col % nx
        y_idx = col // nx
        dhdt_forcing = ATL15["dhdt"][0:nt,y_idx,x_idx].values
        if np.any(np.isnan(dhdt_forcing)):
                dhdt_forcing_int[:,y_idx,x_idx] = np.nan
        else:
                dhdt_forcing_int[:,y_idx,x_idx] = np.interp(model_dates.astype(float), icesat2_dates.astype(float), dhdt_forcing)

for t in range(1,len(model_dates)):
        thk_forcing_int[t, :, :] = thk_forcing_int[t-1, :, :] + dhdt_forcing_int[t-1, :, :]*interp_dt

dhdt_forcing_int = dhdt_forcing_int/year #dhdt is now in m/s

# %% [markdown]
# # TDMA Model

# %%
@jit(nopython=True)
def TDMA(ac,bc,cc,dc):
        #a : left coefficients (i-1)
        #b : center coefficients (i)
        #c : right coefficients (i+1)
        #d : right hand vector 
        #xc : solution at t+1

        nf = len(dc) #number of equations

        #Forward Sweep
        for it in range(1,nf):
            mc = ac[it]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] #Overwrites bc
            dc[it] = dc[it] - mc*dc[it-1] #overwrites d

        #Backsubstitution sweep
        xc = np.empty(nf, dtype=np.float64)
        xc[-1] = dc[-1]/bc[-1]

        for il in range(nf-2,-1,-1):
            xc[il] = (dc[il] - cc[il]*xc[il+1])/bc[il]
        return xc

@jit(nopython=True)
def solve_diffusion(h,C_D,Svec,TB_k,TB_knext):
        #h : head at time t
        #C_D : diffusion coefficient = D*dt/dz**2
        n = len(h)
        #Set up coefficients
        aa = -0.5*C_D*np.ones(n-1, dtype=np.float64)
        bb = (1+C_D)*np.ones(n-1, dtype=np.float64)
        cc = -0.5*C_D*np.ones(n-1, dtype=np.float64)

        #Set up rhs solution
        dd = np.zeros(n-1, dtype=np.float64)
        dd[1:-1] = 0.5*C_D*h[3:] + (1-C_D)*h[2:-1] + 0.5*C_D*h[1:-2] + Svec[2:-1]
        #Top Boundary Condition
        dd[0] = 0.5*C_D*h[2] + (1-C_D)*h[1] + 0.5*C_D*(TB_k + TB_knext)+ Svec[1]


        #Bottom Boundary Condition
        dd[-1] = (1-C_D)*h[-1] + C_D*h[-2] + Svec[-1]
        aa[-1] = -C_D

        # Call Solver
        h[1:] = TDMA(aa,bb,cc,dd)

        return h

@jit(nopython=True, parallel=True)
def run_model(nx,ny,intSave,xi,rhoi,rhow,K_t,D,dt,dz,L,m_nt,h_init):
    n_save = int(m_nt/intSave)+1

    sfc_flux_array = np.zeros((n_save,ny,nx))
    ncols = nx*ny

   

    #############
    for col in prange(ncols):
        x_idx = col % nx
        y_idx = col // nx
        #Parameters
        nz = int(L/dz)+1

        h_all = np.zeros((nz,n_save), dtype=np.float64)
        Svec = np.zeros(nz, dtype=np.float64)

        C_D = D*dt/(dz*dz)

        #Initial head condition is from spinup
        hm = h_init[:,y_idx,x_idx]
        h_all[:,0] = hm

        for k in range(m_nt-1):
            TB_k = (rhoi/rhow)*thk_forcing_int[k,y_idx,x_idx]
            TB_knext = (rhoi/rhow)*thk_forcing_int[k+1,y_idx,x_idx]
            hm[0] = TB_k
            Svec[:] = dt*xi*(rhoi/rhow)*dhdt_forcing_int[k,y_idx,x_idx] #dhdt forcing resides outside function with resolution of dt

            hm = solve_diffusion(hm,C_D,Svec,TB_k,TB_knext)
            hm[0] = TB_knext

            if k%intSave == 0:
                h_all[:,int(k/intSave)] = hm
        
        sfc = K_t*(h_all[0,:]-h_all[1,:])/dz 
        sfc_flux_array[:,y_idx,x_idx] = sfc
        
        
    return sfc_flux_array

# %% [markdown]
# # Run Model

# %%
start = time.time()
sfc_flux_array = run_model(nx,ny,intSave,xi,rhoi,rhow,K_t,D,dt,dz,L,m_nt,h_init)
end = time.time()
print(end-start)

# %% [markdown]
# # Save Output

# %%
time_save = np.arange(0,np.shape(sfc_flux_array)[0],1)

data_q = xr.DataArray(
    sfc_flux_array,
    coords={"time" : time_save, "x" : ATL15["x"].values, "y" : ATL15["y"].values},
    dims=["time","y","x"]
)

data_q.to_netcdf(path + "/Outputs/observation/qflux_CTCK.nc", mode="w", format="NETCDF4")

# %% [markdown]
# # DIAGNOSTIC Plot Movie

# %%
timeind = 0
gif_dates = np.arange(dh.time.values[0], dh.time.values[nt-1], np.timedelta64(5,'D'))

print('now creating gif')
# Setup figure and axis
projection = ccrs.SouthPolarStereo()
fig = plt.figure(figsize=(9, 8))
ax = plt.subplot(1, 1, 1, projection=projection)
# Initialize the plot with first frame
p1 = ax.pcolormesh(ATL15["x"], ATL15["y"], -1000*year*sfc_flux_array[timeind, :, :], 
                   cmap=cmocean.cm.tarn,
                   norm=colors.SymLogNorm(linthresh=1.0, linscale=0.03, vmin=-100, vmax=100))
scripps_gl.plot(ax=ax, edgecolor="black", linewidth=1, facecolor="none")
ax.gridlines(draw_labels=True)
cb = plt.colorbar(p1, ax=ax, orientation='vertical', ticks=[-100, -10, 0, 10, 100])
cb.ax.set_ylabel('Exfiltration Rate (mm/yr)', fontsize=20)
title = plt.title(gif_dates[timeind])

plt.tight_layout()

# Animation update function
def update(timeind):
    # Update the data
    if timeind % 50 == 0:
        print(timeind)
    p1.set_array(-1000*year*sfc_flux_array[timeind, :, :])
    # Update the title
    title.set_text(gif_dates[timeind])

    return p1, title

# Create animation
anim = animation.FuncAnimation(fig, update, frames=range(np.shape(sfc_flux_array)[0]), 
                              interval=25, blit=True)

# Display in Jupyter
#HTML(anim.to_jshtml())

# Or save as GIF
anim.save('25jul_CTCK.gif', writer='pillow', fps=10)
plt.close()



