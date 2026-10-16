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
# ## Load Data

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

bedmachine = xr.open_dataset(path+"/Inputs/BedMachineAntarctica-v3.nc")
bm_x, bm_y = np.meshgrid(bedmachine.x, bedmachine.y)
atl15_x, atl15_y = np.meshgrid(ATL15.x, ATL15.y)
interp = RegularGridInterpolator(points=[bedmachine.x.values,bedmachine.y.values], values=bedmachine["thickness"].values.T, method="linear", fill_value=np.nan, bounds_error = False)
bm_new = interp((atl15_x, atl15_y))

# %%
da = rio.open_rasterio(path+'/Inputs/smith_product.tif')
dhdt = da.variable.data[0]

sm_x, sm_y = np.meshgrid(da[0].x, da[0].y)
atl15_x, atl15_y = np.meshgrid(ATL15.x, ATL15.y)
interp = RegularGridInterpolator(points=[da[0].x.values,da[0].y.values], values=dhdt.T, method="linear", fill_value=np.nan, bounds_error = False)
sm_new = interp((atl15_x, atl15_y))

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
tf = year*16
m_nt = int(np.round(tf/dt)) + 1
# t = np.linspace(0,tf,num=m_nt+1)

time_array = np.linspace(0, 16, m_nt) #16 years. Used to calculated elapsed time, which is then used to calculate new thk

#save each 5 days (10 timesteps)
intSave = 60

# %% [markdown]
# # Forcing

# %%
@jit(nopython=True, parallel=True)
def compute_forcing(nx,ny,n_spec,time_array,dhdt_2019,dt,thk_isnan):
    dhdt_2003 = ((n_spec + 1)/n_spec)*(sm_new - dhdt_2019/(n_spec+1))
    ramp_factor = ((time_array/16.0)**n_spec).astype(np.float64)

    dhdt_forcing_int = np.zeros((m_nt,ny,nx), dtype=np.float64)
    thk_forcing_int  = np.zeros((m_nt, ny, nx), dtype=np.float64)

    interp_dt = dt/year #dt of 0.5 days in year
    ncols = nx*ny

    for col in prange(ncols):
            x_idx = col % nx
            y_idx = col // nx
            if np.isnan(dhdt_2019[y_idx,x_idx]):
                dhdt_forcing_int[:,y_idx,x_idx] = np.nan
            else:
                d03 = dhdt_2003[y_idx,x_idx]
                d19 = dhdt_2019[y_idx,x_idx]
                delta = d19 - d03
                for t in range(m_nt):
                    dhdt_forcing_int[t, y_idx, x_idx] = d03 + delta * ramp_factor[t]

                thk_forcing_int[m_nt-1,y_idx,x_idx] = thk_isnan[y_idx, x_idx]
                for t in range(m_nt-2,-1,-1):
                        thk_forcing_int[t,y_idx,x_idx] = (thk_forcing_int[t+1,y_idx,x_idx] - dhdt_forcing_int[t,y_idx,x_idx]*interp_dt)

    return dhdt_forcing_int,thk_forcing_int

# %%
n_spec = 1.0 #Assumes linear rate of increase in thinning rate

dhdt_2019 = ATL15.dhdt[0,:,:].values
thk_isnan = bm_new
thk_isnan[np.isnan(dhdt_2019)] = np.nan


dhdt_forcing_int,thk_forcing_int = compute_forcing(nx,ny,n_spec,time_array,dhdt_2019,dt,thk_isnan)

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
def run_model(nx,ny,intSave,xi,rhoi,rhow,K_t,D,dt,dz,L,m_nt):
    n_save = int(m_nt/intSave)+1
    sfc_flux_array = np.zeros((n_save,ny,nx))

    z = np.arange(0,L+dz,dz)
    nz = np.size(z)
    h_final = np.zeros((nz,ny,nx))
    ncols = nx*ny

    #############
    for col in prange(ncols):
        x_idx = col % nx
        y_idx = col // nx
        #Parameters
        
        
        h = np.ones(nz, dtype=np.float64)
        h_all = np.zeros((nz,n_save), dtype=np.float64)
        Svec = np.zeros(nz, dtype=np.float64)

        C_D = D*dt/(dz*dz)

        #Initial head condition is scaled to ratio of ice to water densities
        hm = thk_forcing_int[0,y_idx,x_idx]*(rhoi/rhow)*h
        h_all[:,0] = hm

        for k in range(m_nt-1):
            TB_k = (rhoi/rhow)*thk_forcing_int[k,y_idx,x_idx]
            TB_knext = (rhoi/rhow)*thk_forcing_int[k+1,y_idx,x_idx]
            hm[0] = TB_k 
            Svec[:] = dt*xi*(rhoi/rhow)*dhdt_forcing_int[k,y_idx,x_idx]/year 
            hm = solve_diffusion(hm,C_D, Svec,TB_k,TB_knext)
            hm[0] = TB_knext
            if k%intSave == 0:
                h_all[:,int(k/intSave)] = hm
        
        sfc = K_t*(h_all[0,:]-h_all[1,:])/dz 
        sfc_flux_array[:,y_idx,x_idx] = sfc
        h_final[:,y_idx,x_idx] = hm
        
    return sfc_flux_array, h_final

# %% [markdown]
# # Model Run

# %%
start = time.time()
sfc_flux_array, h_final = run_model(nx,ny,intSave,xi,rhoi,rhow,K_t,D,dt,dz,L,m_nt)
end = time.time()
print(end-start)

time_save = np.arange(0,np.shape(sfc_flux_array)[0],1)

data = xr.DataArray(
    h_final,
    coords={"z" : np.arange(0,nz), "x" : ATL15["x"].values, "y" : ATL15["y"].values},
    dims=["z","y","x"]
)
data_q = xr.DataArray(
    sfc_flux_array,
    coords={"time" : time_save, "x" : ATL15["x"].values, "y" : ATL15["y"].values},
    dims=["time","y","x"]
)

# %%
data.to_netcdf(path + "/Outputs/observation/spinup_head_CTCK.nc", mode="w", format="NETCDF4")
data_q.to_netcdf(path + "/Outputs/observation/spinup_qflux_CTCK.nc", mode="w", format="NETCDF4")


