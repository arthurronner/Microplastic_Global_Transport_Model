import netCDF4 as nc
import numpy as np
from multiprocessing import Process
import sys
def get_data_matrix(areas, raster_nc, flow_map, num_basins=53, large_t=True):
    world_mask = areas["mask5minFromTop_map"][:, :].astype(int)
    mp_names = raster_nc.mp_names.copy()
    num_mp = len(mp_names)
    num_tstep = len(raster_nc['time'])
    # Dimensions of the data are [time, stock type, mp type, basin]
    data_arr = np.zeros((num_tstep, 3, num_mp, num_basins))
    print(flow_map.shape)

    # BASINS ARE SUS SED SINKS
    if not large_t:
        temp_sus = np.zeros((num_tstep, num_mp, 2160, 4320))
        temp_sed = np.zeros_like(temp_sus)
        for i in range(num_mp):
            temp_sus[:, i, :, :] = raster_nc[f"sus_mp_{mp_names[i]}"][:, :, :]
            temp_sed[:, i, :, :] = raster_nc[f"sed_mp_{mp_names[i]}"][:, :, :]
        print('done with loading data')
        for j in range(num_basins):
            print(f'starting basin {j}')
            basin_locs = np.where((flow_map == 5) & (world_mask == j+1))
            river_locs = np.where((flow_map != 5) & (world_mask == j+1))

            data_arr[:, 0, :, j] += np.nansum(temp_sus[:, :, river_locs[0], river_locs[1]], axis=2)
            data_arr[:, 2, :, j] += np.nansum(temp_sus[:, :, basin_locs[0], basin_locs[1]], axis=2)

            data_arr[:, 1, :, j] += np.nansum(temp_sed[:, :, river_locs[0], river_locs[1]], axis=2)
            data_arr[:, 2, :, j] += np.nansum(temp_sed[:, :, basin_locs[0], basin_locs[1]], axis=2)
    else:
        for t in range(num_tstep):

            temp_sus = np.zeros((num_mp, 2160, 4320))
            temp_sed = np.zeros_like(temp_sus)
            for i in range(num_mp):
                temp_sus[i, :, :] = raster_nc[f"sus_mp_{mp_names[i]}"][t, :, :]
                temp_sed[i, :, :] = raster_nc[f"sed_mp_{mp_names[i]}"][t, :, :]

            for j in range(num_basins):
                basin_locs = np.where((flow_map == 5) & (world_mask == j + 1))
                river_locs = np.where((flow_map != 5) & (world_mask == j + 1))

                data_arr[t, 0, :, j] += np.nansum(temp_sus[:, river_locs[0], river_locs[1]], axis=1)
                data_arr[t, 2, :, j] += np.nansum(temp_sus[:, basin_locs[0], basin_locs[1]], axis=1)

                data_arr[t, 1, :, j] += np.nansum(temp_sed[:, river_locs[0], river_locs[1]], axis=1)
                data_arr[t, 2, :, j] += np.nansum(temp_sed[:, basin_locs[0], basin_locs[1]], axis=1)

    return data_arr

def run_data_collection(data_file, dir_file, basin_file, save_location):
    output = nc.Dataset(data_file)
    directions = nc.Dataset(dir_file)['lddMap'][:,:]
    basins = nc.Dataset(basin_file)
    start_time = time.time()
    data = get_data_matrix(basins, output, directions, large_t=True)
    end_time = time.time()
    print(f"this took {np.round(end_time - start_time, 3)} seconds")
    #print(data)
    np.save(save_location, data)


if __name__ == '__main__':
    import time
    processes = []
    run_ind = 1
    log_file = open(f'agg_res/log_{run_ind}', 'w')
    sys.stdout = log_file

    for i in range(24):
        proc = Process(target=run_data_collection, args=(
            f'final_outputs/mix_{run_ind}/final_run_{run_ind}_{i}.nc',
            'server_inputs/channel_parameters_extended.nc',
            'server_inputs/mask5minFromTop.nc',
            f'agg_res/mix_{run_ind}/agg_res_{run_ind}_{i}.npy'
        ))
        proc.start()
        processes.append(proc)

    for p in processes:

        p.join()
    log_file.close()
