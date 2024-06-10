import pandas as pd
from samples_util import samples
import numpy as np
from model_discrete_time import MPGTModel, DataPreparationModule, copy_netCDF_vars
import time
import netCDF4 as nc
from multiprocessing import Process



def call_model(ind, row, mps, directory, file_name, T_series_file, Q_series_file, emission_file,
               time_steps, save_int):
    print(f"starting run {ind}")
    em_key = f"test_{ind}"
    # emission_start = time.time()
    # dat = DataPreparationModule(river_in_file=directory + file_name)
    # dat.point_source_emissions_to_netCDF(directory + "HydroWASTE_v10.csv", directory + f"wwtps_small_test_{ind}.nc",
    #                                       out_excel=
    #                                       f"D:\\point_sources_test_{ind}.xlsx",
    #                                      saved_vars=[em_key], inputs=row)
    # emission_end = time.time()
    # print(f"emissions calc took: {np.round(emission_end - emission_start, 3)} seconds")
    # print(row)
    hello = MPGTModel(directory + file_name, "yu", mps, time_steps=time_steps, Q_in_file=Q_series_file,
                      T_in_file=T_series_file,
                      emission_in_file=emission_file, time_chunk=3,
                      save_interval=save_int,
                      dt=1,
                      visc_in_file=directory + "input_mu.xlsx",
                      warm_start=None,
                      inputs=row,
                      print_yu=False, partial_run=False,

                      #emissions_key=em_key,
                      )
    hello.run_model(f"D:\\test2_yu_uncert_0.nc", keep_result_open=False, report_runtime=False)

def MP_parallelisation(n_cores = 8):
    #ToDo: Make sure that this does not just copy vars from outer scope

    mps_list = [None for i in range(n_cores)]
    mp_per_core = int(np.ceil(len(mps_["names"])/n_cores))

    for i in range(len(mps_list)):
        mps_list[i] = {}
        for j in mps_.keys():
            if mp_per_core == 1:
                mps_list[i][j] = np.array([mps_[j][i]])
            else:
                mps_list[i][j] = mps_[j][mp_per_core*i:mp_per_core*(i+1)]
    processes = []
    row_ = df.iloc[0]

    toinclude = ["latitude", "longitude", "time"]
    Q_file = nc.Dataset(Q_series_file_)
    dataset = copy_netCDF_vars(Q_file, "D:\\test2_yu_uncert_0.nc", toinclude,
                               copy_all_time_values=False)
    start_time = Q_file["time"][0]

    saved_timesteps = np.array([start_time + i for i in range(save_interval, dt_ * time_steps_,
                                                                            save_interval)])
    dataset["time"][:] = saved_timesteps

    for i in mps_["names"]:
        dataset.createVariable(f"sed_mp_{i}", "f4", ("time", "latitude", "longitude"), fill_value=0)
        dataset.createVariable(f"sus_mp_{i}", "f4", ("time", "latitude", "longitude"), fill_value=0)

    dataset.close()
    for ind_ in range(len(mps_list)):
        row_ = df.iloc[0]
        print(mps_list[ind_]["names"])
        proc = Process(target=call_model, args=(ind_, row_, mps_list[ind_], directory_, file_name_,
                                                T_series_file_, Q_series_file_, emission_file,
                                                time_steps_, save_interval))
        processes.append(proc)
        proc.start()
        time.sleep(2)

    for p in processes:
        p.join()

if __name__ == "__main__":
    srt_time = time.time()
    sam = samples(input_file='D:\\inputs\\uncert_test.xlsx', scale_dirichlet=True)
    df = pd.DataFrame.from_dict(sam.samples)

    directory_ = "C:\\Users\\Arthur\\Documents\\uni bestanden\\Industrial Ecology\\0. master thesis\\model_input\\"
    file_name_ = "channel_parameters_extended.nc"
    Q_series_file_ = "D:\\inputs\\discharge_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc"
    T_series_file_ = "D:\\inputs\\waterTemp_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc"
    time_steps_ = 12
    hi = DataPreparationModule()
    fac_names = [x for x in df.columns if 'fraction' in x]

    #print(df['wwtps_fiber_fraction'])
    #print(df[fac_names])
    factors = df[fac_names]
    factors = factors.to_dict('records')[0]
    mps_ = hi.create_mp_categories(directory_ + "mp_categories_test_3.xlsx", type_factors=factors)
    print(mps_['names'])
    print(mps_['occurrence'])
    print(np.sum(mps_['occurrence']))
    save_interval = 5
    dt_ = 1
    # SPLIT PER MP CATEGORY
    #MP_parallelisation()

    #SPLIT PER RUN
    processes = []
    for ind_, row_ in df.iterrows():
        proc = Process(target=call_model, args=(ind_,row_, mps_, directory_, file_name_,
                                                          T_series_file_, Q_series_file_, time_steps_, save_interval))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()

    end_tim = time.time()
    print(f"Total runtime for {len(df)} runs of {time_steps_} was {np.round(end_tim-srt_time,3)} seconds")



    print(f"Total runtime for 1 run of {time_steps_} was {np.round(end_tim - srt_time, 3)} seconds")
