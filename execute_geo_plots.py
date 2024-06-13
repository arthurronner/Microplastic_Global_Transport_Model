import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from geo_analysis import load_spatial_totals, plot_chart_map


def plot_totals_pie_chart():
    flow_map = xr.open_dataset('D:\\inputs\\channel_parameters_extended.nc')['lddMap']
    flow_map = flow_map.values
    fig, ax = plt.subplots(figsize=(30, 15))
    data_file = 'D:\\out_tests\\final_local5_0.nc'  # 'D:\\out_tests\\alice_test\\yu_uncert_0.nc'
    names = pd.read_excel("D:\\server_files\\server_inputs\\mp_categories_server_7.xlsx")['names'].tolist()
    geo_data = xr.open_dataset(data_file)
    totals = np.load("D:\\final_outputs\\totals\\mean_total.npy")
    print(totals.shape)
    dat = {
        'sus_mp_total': np.zeros((1, 2160, 4320)),
        'sed_mp_total': np.zeros((1, 2160, 4320))
    }
    dat['sus_mp_total'][0] = totals[0, :, :]
    dat['sed_mp_total'][0] = totals[1, :, :]
    keys_to_plot = ['sus_mp_' + names[i] for i in range(len(names))]
    spat_dat = load_spatial_totals(n_mix=6)
    coefficients = np.nanstd(spat_dat, axis=0) / np.nanmean(spat_dat, axis=0)
    coefficients = np.nan_to_num(coefficients, copy=False)
    print(spat_dat.shape)
    spat_tot = np.nansum(spat_dat, axis=-1)
    global_values = np.zeros((2, 4))
    global_values[0, :] = np.nanmean(spat_tot[:, :], axis=0)
    global_values[1, :] = np.nanstd(spat_tot[:, :], axis=0) / np.nanmean(spat_tot[:, :], axis=0)
    plot_chart_map("D:\\world_basins\\pcr_basins\\mask5minFromTop.nc", dat,
                   ['total'], keys_to_plot, flow_map, ax, land_type='raster', chart_type='pie_stocks',
                   type_names=["Suspended", "Sedimented", "Exported"],
                   colors=["cornflowerblue", 'orange', "mediumorchid", "purple"],
                   plot_variances=None,
                   plot_total=global_values)
    ax.set_xlim(-181, 181)
    ax.set_ylim(-70, 91)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.gca().set_position([-0.5, 0, 2, 1])
    plt.savefig(f'D:\\final_outputs\\stock_map.png', dpi=400)
    plt.show()

def plot_variances_map(n_mix=8):

    flow_map = xr.open_dataset('D:\\inputs\\channel_parameters_extended.nc')['lddMap']
    flow_map = flow_map.values
    fig, ax = plt.subplots((2,2), figsize=(40, 27))
    data_file = 'D:\\out_tests\\final_local5_0.nc'  # 'D:\\out_tests\\alice_test\\yu_uncert_0.nc'
    names = pd.read_excel("D:\\server_files\\server_inputs\\mp_categories_server_7.xlsx")['names'].tolist()
    geo_data = xr.open_dataset(data_file)
    totals = np.load("D:\\final_outputs\\totals\\mean_total.npy")
    print(totals.shape)
    dat = {
        'sus_mp_total': np.zeros((1, 2160, 4320)),
        'sed_mp_total': np.zeros((1, 2160, 4320))
    }
    dat['sus_mp_total'][0] = totals[0, :, :]
    dat['sed_mp_total'][0] = totals[1, :, :]
    keys_to_plot = ['sus_mp_' + names[i] for i in range(len(names))]
    spat_dat = load_spatial_totals(n_mix=n_mix)
    coefficients = np.nanstd(spat_dat, axis=0) / np.nanmean(spat_dat, axis=0)
    coefficients = np.nan_to_num(coefficients, copy=False)
    print(spat_dat.shape)
    spat_tot = np.nansum(spat_dat, axis=-1)
    global_values = np.zeros((2, 4))
    global_values[0, :] = np.nanmean(spat_tot[:, :], axis=0)
    global_values[1, :] = np.nanstd(spat_tot[:, :], axis=0) / np.nanmean(spat_tot[:, :], axis=0)
    plot_chart_map("D:\\world_basins\\pcr_basins\\mask5minFromTop.nc", dat,
                   ['total'], keys_to_plot, flow_map, ax, land_type='raster', chart_type='pie_stocks',
                   type_names=["Suspended", "Sedimented", "Exported"],
                   colors=["cornflowerblue", 'orange', "mediumorchid", "purple"],
                   plot_variances=None,
                   plot_total=global_values)
    ax.set_xlim(-181, 181)
    ax.set_ylim(-70, 86)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.gca().set_position([-0.5, 0, 2, 1])
    plt.savefig(f'D:\\final_outputs\\test_stock_map.png', dpi=400)
    plt.show()

if __name__ == '__main__':
    plot_totals_pie_chart()