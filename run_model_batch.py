import pandas as pd
import numpy as np
from model_discrete_time import MPGTModel, DataPreparationModule
import time
from multiprocessing import Process, Semaphore

def run_batch(input_directory, output_folder, wwtps_out_file_name,
              river_file_name="channel_parameters_extended.nc",
              q_file_name="discharge_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc",
              temp_file_name="waterTemp_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc",
              uncertainty_file_name="uncertainties.xlsx",
              sample_file_name="sample.xlsx",
              washing_file_name="washing_data.xlsx",
              mp_settings_file_name="mp_categories_settings.xlsx",
              mp_categories_file_name="mp_categories_test_3.xlsx",
              wwtps_file_name="HydroWASTE_v10.csv",
              wwtps_excel_name='point_sources',
              visc_file_name="visc_table.xlsx",
              output_file_name="final_test",
              create_mp_file=True,
              time_steps=4,
              model_dt=1,
              model_saveint=2,
              model_timechunk=3,
              test_run=False,
              parallel=False,
              sample_exists=False,
              max_proces=None
              ):


    df = pd.read_excel(input_directory+sample_file_name)

    # directory = "C:\\Users\\Arthur\\Documents\\uni bestanden\\Industrial Ecology\\0. master thesis\\model_input\\"
    river_file = input_directory + river_file_name
    q_series_file = input_directory + q_file_name
    temp_series_file = input_directory + temp_file_name
    washing_file = input_directory + washing_file_name
    wwtps_file = input_directory + wwtps_file_name

    fac_names = [x for x in df.columns if 'fraction' in x]
    if type(mp_categories_file_name) is not str:
        num_mp_mixes = len(mp_categories_file_name)
    else:
        num_mp_mixes = 1

    processes = []
    if max_proces is not None:
        sema = Semaphore(max_proces)
    else:
        sema = None

    for j in range(num_mp_mixes):
        if num_mp_mixes > 1:
            mp_cat_file = mp_categories_file_name[j]
        else:
            mp_cat_file = mp_categories_file_name
        for ind, row in df.iterrows():
            args = (ind, row, input_directory, output_folder, wwtps_out_file_name)
            kwargs = {
                'river_file': river_file,
                'q_series_file': q_series_file,
                'temp_series_file': temp_series_file,
                'mp_categories_file_name': mp_cat_file,
                'wwtps_file': wwtps_file,
                'washing_file': washing_file,
                'wwtps_excel_name': wwtps_excel_name,
                'visc_file_name': visc_file_name,
                'output_file_name': output_file_name,
                'time_steps': time_steps,
                'pol_ind': j,
                'model_dt': model_dt,
                'model_saveint': model_saveint,
                'model_timechunk': model_timechunk,
                'test_run': test_run,
                'fac_names': fac_names,
                "sema": sema
            }
            if not parallel:
                model_call(*args,**kwargs)
            else:
                sema.acquire()
                proc = Process(target=model_call, args=args, kwargs=kwargs)
                proc.start()
                processes.append(proc)



    if parallel:
        for p in processes:
            p.join()


def model_call(ind, row, input_directory, output_folder, wwtps_out_file_name,
               river_file="channel_parameters_extended.nc",
               q_series_file="discharge_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc",
               temp_series_file="waterTemp_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc",
               mp_categories_file_name="mp_categories_test_3.xlsx",
               washing_file="washing_data.xlsx",
               wwtps_file="HydroWASTE_v10.csv",
               wwtps_excel_name='point_sources',
               visc_file_name="visc_table.xlsx",
               output_file_name="final_test",
               pol_ind=0,
               time_steps=4,
               model_dt=1,
               model_saveint=2,
               model_timechunk=3,
               test_run=False,
               fac_names=[],
               sema=None,
               ):

    em_key = f"test_{ind}"
    emission_start = time.time()
    dat = DataPreparationModule(river_in_file=river_file)
    wwtps_nc_file = output_folder + f"{wwtps_out_file_name}_{ind}.nc"
    dat.point_source_emissions_to_netCDF(wwtps_file, wwtps_nc_file,
                                         save_excel='data_only',
                                         out_excel=output_folder + f"{wwtps_excel_name}_{pol_ind}_{ind}.xlsx",
                                         saved_vars=[em_key], inputs=row, washing_file=washing_file)
    emission_end = time.time()

    # print(f"current memory use: {tracemalloc.get_traced_memory()}")
    print(f"emissions calc took: {np.round(emission_end-emission_start, 3)} seconds")

    factors = row[fac_names]
    factors = factors.to_dict()
    mps_ = dat.create_mp_categories(input_directory + mp_categories_file_name, type_factors=factors)

    model = MPGTModel(river_file, "yu", mps_, time_steps=time_steps, Q_in_file=q_series_file,
                      T_in_file=temp_series_file,
                      emission_in_file=wwtps_nc_file, time_chunk=model_timechunk,
                      save_interval=model_saveint,
                      dt=model_dt,
                      visc_in_file=input_directory + visc_file_name,
                      warm_start=None,
                      inputs=row,
                      emissions_key=em_key,
                      compression_level=-3,
                      )

    model.run_model(output_folder+f"{output_file_name}_{ind}.nc", keep_result_open=False, report_runtime=False)
    # print(f"current memory use: {tracemalloc.get_traced_memory()}")
    hello = None
    if sema is not None:
        sema.release()
    if test_run:
        return


if __name__ == '__main__':
    # tracemalloc.start()
    print('hello we starting')
    run_batch("D:\\Arthur\\Documents\\Thesis\\inputs\\",
              "D:\\Arthur\\Documents\\Thesis\\outputs\\", "wwtps_emissions",
              output_file_name="final_local5",
              mp_categories_file_name=[f"mp_categories_server_{j}.xlsx" for j in range(2)],
              sample_file_name='final_sample5.xlsx',
              time_steps=1825,
              model_saveint=91,
              model_timechunk=2,
              #max_proces=2,
              parallel=False,

              )
    # print(f"current memory use: {tracemalloc.get_traced_memory()}")
    # tracemalloc.stop()