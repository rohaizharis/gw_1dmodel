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
#from datetime import date, timedelta
import xarray as xr
from xarray.coders import CFDatetimeCoder

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
import copy
import sys
# Experiment Parameters

model_name = sys.argv[1]
experiment_name = "expAE10"
# # Load Data
print(f"Experiment = {experiment_name}, Model = {model_name}")

path = os.path.dirname(os.path.dirname(os.getcwd())) + "/scratch"
print(path)
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

da = rio.open_rasterio(path+'/Inputs/smith_product.tif')
dhdt = da.variable.data[0]

sm_x, sm_y = np.meshgrid(da[0].x, da[0].y)
mod_x, mod_y = np.meshgrid(lithk.x, lithk.y)
interp = RegularGridInterpolator(points=[da[0].x.values,da[0].y.values], values=dhdt.T, method="linear", fill_value=np.nan, bounds_error = False)
sm_new = interp((mod_x, mod_y))

ssb_x, ssb_y = np.meshgrid(ssb.x, ssb.y)
ssb_val = ssb.values
ssb_val = ssb_val[0,:,:]
ssb_val[ssb_val < -9000] = np.nan

interp = RegularGridInterpolator(points=[ssb.x.values,ssb.y.values], values=ssb_val.T, method="nearest", fill_value=np.nan, bounds_error = False)
ssb_new = interp((mod_x, mod_y))

perm_var = np.zeros_like(ssb_new)
perm_var[ssb_new == 3] = 1e-13      #T1 Basin
perm_var[ssb_new == 2] = 1e-13      #T2 Basin
perm_var[ssb_new == 1] = 1e-13          #Intrabasin volcanics 
perm_var[ssb_new == 0] = 1e-21      #Crystalline
perm_var[ssb_new == -1] = 1e-17     #T2 + C
perm_var[ssb_new == -2] = 1e-18     #T1 and C
perm_var[ssb_new == -3] = 1e-13     #T1 and T2

# # Parameters

year = 3600*24*365.25
nt,ny,nx = np.shape(dlithkdt["dlithkdt"].values)

dz = 100
dt = 3600*2
tf = year*16
m_nt = int(np.round(tf/dt) + 1)
t = np.linspace(0,tf,num=m_nt-1)

time_array = np.linspace(0, 16, m_nt) #16 years. Used to calculated elapsed time, which is then used to calculate new thk

#save each 5 days (10 timesteps)
intSave = 60

# # Varying Thickness

thk = xr.open_dataset(path+"/Inputs/Ant_Crust.nc")
thk.rio.write_crs("epsg:3031", inplace=True)
thk = thk.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs)

thk_x, thk_y = np.meshgrid(thk.x, thk.y)
interp = RegularGridInterpolator(points=[thk.x.values,thk.y.values], values=thk["Mean_SSB_th"].values.T, method="linear", fill_value=np.nan, bounds_error = False)
thk_new = interp((mod_x,mod_y))

thk_threshold = copy.deepcopy(thk_new)
thk_threshold[thk_threshold < 300] = np.nan
thk_discr = np.round(thk_threshold / dz) * dz
max_L = thk_discr[~np.isnan(thk_discr)].max()
max_nz = int(max_L/dz) + 2

# # Forcing


n_spec = 1.0 #Assumes linear rate of increase in thinning rate

dhdt_2019 = dlithkdt["dlithkdt"][4,:,:].values*year #Conversion from m/s to m/yr
thk_isnan = lithk["lithk"][3,:,:].values

dhdt_2003 = ((n_spec + 1)/n_spec)*(sm_new - dhdt_2019/(n_spec+1))
ramp_factor = ((time_array/16.0)**n_spec).astype(np.float64) #year/year
delta = dhdt_2019 - dhdt_2003
interp_dt = dt/year #dt of 0.5 days in year

@jit(nopython=True, parallel=True)
def compute_thk_init(nx, ny, n_spec, time_array, dhdt_2019, dt, thk_isnan, sm_new, m_nt):
    year = 3600*24*365.25
    dhdt_2003 = ((n_spec + 1)/n_spec)*(sm_new - dhdt_2019/(n_spec+1))
    ramp_factor = (time_array/16.0)**n_spec
    interp_dt = dt/year

    # Backward integration covered t = 0 .. m_nt-2  →  m_nt-1 terms
    n_steps = m_nt - 1
    ramp_sum = np.sum(ramp_factor[:n_steps])

    thk_forcing_init = np.zeros((ny, nx), dtype=np.float64)
    ncols = nx*ny
    for col in prange(ncols):
        x_idx = col % nx
        y_idx = col // nx
        if np.isnan(dhdt_2019[y_idx, x_idx]):
            thk_forcing_init[y_idx, x_idx] = np.nan   # see note below
        else:
            d03   = dhdt_2003[y_idx, x_idx]
            delta = dhdt_2019[y_idx, x_idx] - d03
            integral = n_steps*d03 + delta*ramp_sum
            thk_forcing_init[y_idx, x_idx] = thk_isnan[y_idx, x_idx] - interp_dt*integral
    return thk_forcing_init

thk_forcing_init = compute_thk_init(nx, ny, n_spec, time_array, dhdt_2019, dt, thk_isnan, sm_new, m_nt)

del libmassbfgr, dlithkdt, lithk, sm_new, ssb_new,thk_isnan
gc.collect()

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

@jit(nopython=True, parallel=True, cache=False)
def run_model(nx,ny,dt,dz,m_nt,max_nz,perm_arr,solve_cols_sorted):
    h_final = np.zeros((max_nz,ny,nx), dtype=np.float64) #NOTE: only h_final grid size is preallocated
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
        if np.isnan(thk_discr[y_idx, x_idx]) or np.isnan(thk_forcing_init[y_idx, x_idx]):#if sediment thickness or bedmachine is NaN (no sed/ice or too thin), set head to NaN
            h_final[:, y_idx, x_idx] = np.nan
    print("Now Solving")
    # Parallelize ONLY the real columns
    n_solve = len(solve_cols_sorted)
    for i in prange(n_solve):
        col   = solve_cols_sorted[i]
        x_idx = col % nx
        y_idx = col // nx

        # Variables initialized here with dimension nz will have varying grid sizes depending on sediment thickness
        L = thk_discr[y_idx,x_idx]
        nz = int(L/dz) + 2
        h = np.ones(nz, dtype=np.float64)
        Svec = np.zeros(nz, dtype=np.float64)

        perm = perm_arr[y_idx,x_idx]
        K_t = perm*rhow*g/mu_w
        D = K_t/S
        C_D = D*dt/(dz*dz)
        thk_m = thk_forcing_init[y_idx,x_idx]
        #Initial head condition is scaled to ratio of ice to water densities
        hm = thk_m*(rhoi/rhow)*h

        for k in range(m_nt-1):
            TB_k = (rhoi/rhow)*thk_m
            dhdt_forcing_int = dhdt_2003[y_idx, x_idx] + delta[y_idx, x_idx] * ramp_factor[k]
            TB_knext = (rhoi/rhow)*(thk_m + dt*dhdt_forcing_int/year) #conversion of m/yr back to m/s
            hm[0] = TB_k
            Svec[:] = dt*xi*(rhoi/rhow)*dhdt_forcing_int/year #conversion of m/yr back to m/s
            hm = solve_diffusion(hm,C_D,Svec,TB_k,TB_knext)
            hm[0] = TB_knext
            thk_m = thk_m + dt*dhdt_forcing_int/year #conversion of m/yr back to m/s

        h_final[0:nz,y_idx,x_idx] = hm.astype(np.float64)
            
    return h_final

# # Run Model
solve_mask = (~np.isnan(thk_discr) & ~np.isnan(thk_forcing_init)).ravel()
solve_cols = np.where(solve_mask)[0].astype(np.int64)
solve_cost = thk_discr.ravel()[solve_cols]
sort_idx = np.argsort(solve_cost)[::-1]
solve_cols = solve_cols[sort_idx].astype(np.int64)

print("TDMA MODEL START")
start = time.time()
h_final = run_model(nx,ny,dt,dz,m_nt,max_nz,perm_var,solve_cols)
end = time.time()
print(end-start)

# # Final Timestep Plot
lithk = xr.open_dataset(path_lithk, decode_times=CFDatetimeCoder(use_cftime=True))
lithk = lithk.drop_vars(["time_bnds", "time_bounds"], errors="ignore")
lithk.rio.write_crs("epsg:3031", inplace=True)
lithk = lithk.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs)

data = xr.DataArray(
    h_final,
    coords={"z" : np.arange(0,max_nz), "x" : lithk["x"].values, "y" : lithk["y"].values},
    dims=["z","y","x"]
)


data.to_netcdf(path+"/Outputs/projection/"+experiment_name+"/spinup_head_"+experiment_name+"_"+model_name+".nc", mode="w", format="NETCDF4")


