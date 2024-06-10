import netCDF4 as nc
import numpy as np

if __name__ == "__main__":
    mixes = [0,1,2,3]
    num_runs = 24
    time_step = 19
    total_dat = np.zeros((2, len(mixes)*num_runs, 2160, 4320), dtype=np.float32)
    for mp in range(15):
        ind = 0
        for i in mixes:
            for j in range(num_runs):
                data = nc.Dataset(f"final_outputs/mix_{i}/final_run_{i}_{j}.nc", mode='r')
                mp_names = data.mp_names
                total_dat[0,ind,:,:] += np.ma.getdata(data[f'sus_mp_{mp_names[mp]}'][time_step,:,:])
                total_dat[1,ind,:,:] += np.ma.getdata(data[f'sed_mp_{mp_names[mp]}'][time_step,:,:])
                ind += 1
                data.close()

    mean_data = np.nanmean(total_dat, axis=1)
    var_data = np.nanvar(total_dat, axis=1)
    np.save('final_outputs/mean_total.npy', mean_data)
    np.save('final_outputs/var_total.npy', mean_data)
