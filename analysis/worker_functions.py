#Worker function script to calculate gw and lmbf sums for projection period runs. Script processes one model and scenario and can be used in parallel

import numpy as np
import xarray as xr
import gc
from xarray.coding.times import CFDatetimeCoder
from shapely.geometry import mapping
import os 
import geopandas 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")

path = "/home/m10936923/scratch"
IMBIE = geopandas.read_file(path+"/Inputs/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp", crs="epsg:3031")
scripps = geopandas.read_file(path+"/Inputs/scripps_GL/scripps_antarctica_polygons_v1.shp", crs="epsg:3031")
scripps_gl = scripps.loc[scripps['Id'].astype(int) == 1]
model_names = ["DOE_MALI",
               "ILTS_SICOPOLIS",
               "NCAR_CISM2",
               "PIK_PISM",
               "UCM_Yelmo",
               "ULB_fETISh-KoriBU1",
               "UTAS_ElmerIce",
               "VUW_PISM1"]


def process_single_model(args):
    bas, exp_name, i = args

    if model_names[i] == "VUW_PISM1":
        sign = 1
    else:
        sign = -1

    gw_path = path + "/Outputs/projection/" + exp_name + "/qflux_" + exp_name + "_" + model_names[i] + ".nc"
    lmbf_path = path + "/Inputs/ISMIP6_all/" + exp_name + "/libmassbfgr/libmassbfgr_AIS_" + model_names[i] + "_" + exp_name + ".nc"

    gw_data = xr.open_dataarray(
            gw_path,
            decode_times=CFDatetimeCoder(use_cftime=True),
            lock=False
        )
    gw_data.rio.write_crs("epsg:3031", inplace=True)
    gw_data = gw_data.rio.clip([IMBIE.loc[bas].geometry], IMBIE.crs, drop=True)
    gw_data = gw_data.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs, drop=True)
    lmbf = xr.open_dataset(
        lmbf_path,
        decode_times=CFDatetimeCoder(use_cftime=True),
        lock=False,
    )

    lmbf = lmbf["libmassbfgr"]
    lmbf.rio.write_crs("epsg:3031", inplace=True)

    lmbf = lmbf.rio.clip([IMBIE.loc[bas].geometry], IMBIE.crs, drop=True)
    lmbf = lmbf.rio.clip(scripps_gl.geometry.apply(mapping), scripps_gl.crs, drop=True)

    lmbf.rio.write_crs("epsg:3031", inplace=True)
    gw_data.rio.write_crs("epsg:3031", inplace=True)


    dx = float(gw_data.x[2] - gw_data.x[1]) 

    basin_gw = -1000*gw_data.values*3600*24*365.25  # convert to mm yr-1 units
    basin_lmbf = sign*lmbf.values*3600*24*365.25  # convert to kg m-2 yr-1 units 

    
    gw_sums   = np.nansum(basin_gw   * (dx**2), axis=(1, 2)) / 1e12  #Gt yr-1
    lmbf_sums = np.nansum(basin_lmbf * (dx**2), axis=(1, 2)) / 1e12  #Gt yr-1

    #Derive gw time-series length from the data . Note, timestep of 65 days used.
    n_gw = gw_sums.shape[0]
    year_indices = (np.arange(n_gw) * 65/365.25).astype(int) + 4

    #Guard against running past the end of the annual lmbf record
    if year_indices.max() >= lmbf_sums.shape[0]:
        raise ValueError(
            f"year_indices max {year_indices.max()} >= lmbf length "
            f"{lmbf_sums.shape[0]} for basin={bas}, exp={exp_name}, "
            f"model={model_names[i]}"
        )

    basin_frac = gw_sums / (gw_sums + lmbf_sums[year_indices])
    
    gw_data.close()
    lmbf.close()
    del gw_data,lmbf
    gc.collect()
    return i,basin_frac,gw_sums, lmbf_sums
