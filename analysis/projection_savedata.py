import os
import numpy as np
import pandas as pd
import geopandas
from concurrent.futures import ProcessPoolExecutor, as_completed
from worker_functions import process_single_model
import multiprocessing as mp


if __name__ == '__main__':
    #edit below lines to match HPC/local machine
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    n_workers = min(n_cores, 12)
    path = "/home/m10936923/scratch"
    IMBIE = geopandas.read_file(path+"/Inputs/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp", crs="epsg:3031")

    exp_names = ["expAE02", "expAE03", "expAE04", "expAE05", "expAE10", "ctrlAE"]
    basin_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18]
    model_names = ["DOE_MALI",
                   "ILTS_SICOPOLIS",
                   "NCAR_CISM2",
                   "PIK_PISM",
                   "UCM_Yelmo",
                   "ULB_fETISH-KoriBU1",
                   "UTAS_ElmerIce",
                   "VUW_PISM1"]
    n_models = 8

    output_path = path + "/Outputs/projection_data_unmasked.parquet"

    tasks = [(bas, exp_name, i)
             for bas in basin_ids
             for exp_name in exp_names
             for i in range(n_models)]

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_model, t): t for t in tasks}
        for future in as_completed(futures):
            bas, exp_name, i = futures[future]
            _, basin_frac, gw_sums, lmbf_sums = future.result()
            results[(bas, exp_name, i)] = (basin_frac, gw_sums, lmbf_sums)

    rows = []
    for bas in basin_ids:
        for exp_name in exp_names:
            for i in range(n_models):
                basin_frac, gw_sums, lmbf_sums = results[(bas, exp_name, i)]
                n_gw_timesteps   = len(gw_sums)
                n_lmbf_timesteps = len(lmbf_sums)

                for t in range(n_gw_timesteps):
                    rows.append({
                        "basin":      bas,
                        "exp_name":   exp_name,
                        "model":      model_names[i],
                        "timestep":   t,
                        "array_type": "gw",
                        "frac":       basin_frac[t],
                        "gw_sums":    gw_sums[t],
                        "lmbf_sums":  np.nan,  #lmbf is different length array, stored separately. Code block below
                    })

                for t in range(n_lmbf_timesteps):
                    rows.append({
                        "basin":      bas,
                        "exp_name":   exp_name,
                        "model":      model_names[i],
                        "timestep":   t,
                        "array_type": "lmbf",
                        "frac":       np.nan,
                        "gw_sums":    np.nan,
                        "lmbf_sums":  lmbf_sums[t],
                    })

            print(f"Saved basin {bas}, {exp_name}")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Written to {output_path}")
    print(df.head())
    print(df.dtypes)