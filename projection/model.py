import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["NUMBA_THREADING_LAYER"] = "tbb"

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
import xarray as xr
from xarray.coders import CFDatetimeCoder
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
import copy
import cftime
import sys

# Experiment Parameters
model_name = sys.argv[1]
experiment_name = "expAE10"
print(f"Experiment = {experiment_name}, Model = {model_name}")

# # Load Data
path = os.path.dirname(os.path.dirname(os.getcwd())) + "/scratch"


IMBIE = geopandas.read_file(path+"/Inputs/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp", crs="epsg:3031")
scripps = geopandas.read_file(path+"/Inputs/scripps_GL/scripps_antarctica_polygons_v1.shp", crs="epsg:3031")
scripps_gl = scripps.loc[scripps['Id'].astype(int) == 1]
ssb = rio.open_rasterio(path+"/Inputs/AntarcticBasins_BedTypeCode.tif")

path_dlithkdt = path+"/Inputs/ISMIP6_all/"+experiment_name+"/dlithkdt/dlithkdt_AIS_"+model_name+"_"+experiment_name+".nc"
path_lithk = path+"/Inputs/ISMIP6_all/"+experiment_name+"/lithk/lithk_AIS_"+model_name+"_"+experiment_name+".nc"
path_libmassbfgr = path+"/Inputs/ISMIP6_all/"+experiment_name+"/libmassbfgr/libmassbfgr_AIS_"+model_name+"_"+experiment_name+".nc"
dlithkdt = xr.open_dataset(path_dlithkdt, decode_times=CFDatetimeCoder(use_cftime=True))
dlithkdt = dlithkdt.drop_vars(["time_bnds", "time_bounds"], errors="ignore")
dlithkdt.rio.write_crs("epsg:3031", inplace=True)
dlithkdt = dlithkdt.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs)

lithk = xr.open_dataset(path_lithk, decode_times=CFDatetimeCoder(use_cftime=True))
lithk = lithk.drop_vars(["time_bnds", "time_bounds"], errors="ignore")
lithk.rio.write_crs("epsg:3031", inplace=True)
lithk = lithk.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs)

libmassbfgr = xr.open_dataset(path_libmassbfgr, decode_times=CFDatetimeCoder(use_cftime=True))
libmassbfgr = libmassbfgr.drop_vars(["time_bnds", "time_bounds"], errors="ignore")
libmassbfgr.rio.write_crs("epsg:3031", inplace=True)
libmassbfgr = libmassbfgr.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs)

ssb_x, ssb_y = np.meshgrid(ssb.x, ssb.y)
mod_x, mod_y = np.meshgrid(lithk.x, lithk.y)
ssb_val = ssb.values
ssb_val = ssb_val[0,:,:]
ssb_val[ssb_val < -9000] = np.nan

interp = RegularGridInterpolator(points=[ssb.x.values,ssb.y.values], values=ssb_val.T, method="nearest", fill_value=np.nan, bounds_error = False)
ssb_new = interp((mod_x, mod_y))

perm_var = np.zeros_like(ssb_new)
perm_var[ssb_new == 3] = 1e-13      #T1 Basin
perm_var[ssb_new == 2] = 1e-13      #T2 Basin
perm_var[ssb_new == 1] = 1e-13      #Intrabasin volcanics 
perm_var[ssb_new == 0] = 1e-21      #Crystalline
perm_var[ssb_new == -1] = 1e-17     #T2 + C
perm_var[ssb_new == -2] = 1e-18     #T1 and C
perm_var[ssb_new == -3] = 1e-13     #T1 and T2

nt,ny,nx = np.shape(dlithkdt["dlithkdt"].values)

year = 3600*24*365.25

dz = 100
dt = 3600*2
tf = year*281
m_nt = int(np.round(tf/dt) + 1)
t = np.linspace(0,tf,num=m_nt-1)

#save each 780 timesteps (65 days)
intSave = 780

h_init_xr = xr.open_dataarray(path+"/Outputs/projection/"+experiment_name+"/spinup_head_"+experiment_name+"_"+model_name+".nc")
h_init = h_init_xr.values

# # Varying Thickness
thk = xr.open_dataset(path+"/Inputs/Ant_Crust.nc")
thk.rio.write_crs("epsg:3031", inplace=True)
thk = thk.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs)

sthk_x, sthk_y = np.meshgrid(thk.x, thk.y)
interp = RegularGridInterpolator(points=[thk.x.values,thk.y.values], values=thk["Mean_SSB_th"].values.T, method="linear", fill_value=np.nan, bounds_error = False)
thk_new = interp((mod_x, mod_y))

thk_threshold = copy.deepcopy(thk_new)
thk_threshold[thk_threshold < 300] = np.nan
thk_discr = np.round(thk_threshold / dz) * dz

# # TDMA Model
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
def run_model(nx,ny,intSave,dt,dz,m_nt,h_init,perm_arr,dhdt,lithk,solve_cols):
    n_save = int(m_nt/intSave)+1
    sfc_flux_array = np.zeros((n_save,ny,nx))
    ncols = nx*ny

    xi = 0.2 #Skempton's constant
    rhoi = 920
    rhow = 1000
    g = 9.81
    mu_w = 1e-3
    S = 1e-6
   
    for col in range(ncols):
        x_idx = col % nx
        y_idx = col // nx
        if np.isnan(thk_discr[y_idx, x_idx]) or np.isnan(dhdt[0,y_idx,x_idx]):#if sediment thickness or bedmachine is NaN (no sed/ice or too thin), set head to NaN
            sfc_flux_array[:,y_idx,x_idx] = np.nan
    print("Now Solving")
    # Parallelize ONLY the real columns
    n_solve = len(solve_cols)
    for i in prange(n_solve):
        col   = solve_cols[i]
        x_idx = col % nx
        y_idx = col // nx

        #Parameters
        # Variables initialized here with dimension nz will have varying grid sizes depending on sediment thickness
        L = thk_discr[y_idx,x_idx]
        nz = int(L/dz) + 2
        h_all = np.zeros((nz,n_save), dtype=np.float64)
        Svec = np.zeros(nz, dtype=np.float64)
        perm = perm_arr[y_idx,x_idx]
        K_t = perm*rhow*g/mu_w
        D = K_t/S
        C_D = D*dt/(dz*dz)
        #Initial head condition is from spinup
        thk_m = lithk[3,y_idx,x_idx]
        hm = h_init[0:nz,y_idx,x_idx].copy()
        h_all[:,0] = hm
        for k in range(m_nt-1):
            year_idx = int(k * dt / year) + 4

            TB_k = (rhoi/rhow)*thk_m
            TB_knext = (rhoi/rhow)*(thk_m + dt*dhdt[year_idx,y_idx,x_idx])



            hm[0] = TB_k
            Svec[:] = dt*xi*(rhoi/rhow)*dhdt[year_idx,y_idx,x_idx]

            hm = solve_diffusion(hm,C_D,Svec,TB_k,TB_knext)
            hm[0] = TB_knext
            thk_m = thk_m + dt*dhdt[year_idx,y_idx,x_idx]
            if k%intSave == 0:
                h_all[:,int(k/intSave)] = hm
        
        sfc = K_t*(h_all[0,:]-h_all[1,:])/dz 
        sfc_flux_array[:,y_idx,x_idx] = sfc
        
    return sfc_flux_array

# # Run Model
solve_mask = (~np.isnan(thk_discr) & ~np.isnan(dlithkdt["dlithkdt"][0,:,:].values)).ravel()
solve_cols = np.where(solve_mask)[0].astype(np.int64)
solve_cost = thk_discr.ravel()[solve_cols]
sort_idx = np.argsort(solve_cost)[::-1]
solve_cols = solve_cols[sort_idx].astype(np.int64)

print("TDMA MODEL START")
start = time.time()
sfc_flux_array = run_model(nx,ny,intSave,dt,dz,m_nt,h_init,perm_var,dlithkdt["dlithkdt"].values,lithk["lithk"].values,solve_cols)
end = time.time()
print(end-start)

# # Save Model
time_save = np.arange(0,np.shape(sfc_flux_array)[0],1)

data_q = xr.DataArray(
    sfc_flux_array,
    coords={"time" : time_save, "x" : lithk["x"].values, "y" : lithk["y"].values},
    dims=["time","y","x"]
)

data_q.to_netcdf(path+"/Outputs/projection/"+experiment_name+"/qflux_"+experiment_name+"_"+model_name+".nc", mode="w", format="NETCDF4")

