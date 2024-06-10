
import netCDF4 as nc
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import numpy as np
import pickle
from cftime import num2date, date2num
from matplotlib.colors import LogNorm
import seaborn as sns
import distinctipy as dp
import pingouin as pg

#output_file = "D:\\test_run_new_mass_balance.nc"

#data = nc.Dataset(output_file, "r")
def save_mp_tables(n_mix = 8):
    from model_discrete_time import DataPreparationModule
    for mix_ind in range(n_mix):
        file = f"D:\\server_files\\server_inputs\\mp_categories_server_{mix_ind}.xlsx"
        mps = pd.read_excel(file)
        print(mps['names'])
        letters = ['', '','', 'D','E', 'F','G', 'H']
        dat_prep = DataPreparationModule()
        type_names =[f"wwtps_{i}_fraction" for i in ['fiber', 'fragment', 'bead', 'film', 'foam']]
        dic = {}
        for i in type_names:
            dic[i] = 1
        print(mps.columns)
        outcomes = dat_prep.create_mp_categories(file, type_factors=dic)
        mps['occurrence'] = outcomes['occurrence']
        print(mps[['alow', 'a', 'aupp']])
        mps = mps[['names', 'density', 'alow', 'a', 'aupp', 'b', 'c', 'CSF', 'sphericity', 'volume', 'occurrence']]
        use_cols = ['names', 'density', 'alow', 'a', 'aupp', 'b', 'c', 'CSF', 'sphericity', 'volume', 'occurrence']
        use_cols_alias = ['col1names', 'col2density', 'col3alow', 'col4a', 'col5aupp', 'col6b', 'col7c', 'col8CSF', 'col9sphericity', 'col10volume', 'col11occurrence']
        col_names = ['Names', r'$\rho_s$ \newline (\si{kg})', r'$a_{low}$ \newline (\si{mm})', r'$a$  \newline (\si{mm})',
                             r'$a_{upp}$ \newline (\si{mm})', r'$b$ \newline (\si{mm})',r'$c$ \newline (\si{mm})','$CSF$',
                             '$\psi$', r'$V_p$ ($\times 10^{-3}$ \newline  \si{mm^3})', 'Occurrence']
        lat = mps.to_latex(
                        columns=use_cols,
                        header=use_cols_alias,
                        formatters={'alow':lambda x: r'\num{'+ f'{x*1e3:.3g}' + r'}',
                                    'a':lambda x: r'\num{'+ f'{x*1e3:.3g}' + r'}',
                                    'aupp':lambda x: r'\num{'+ f'{x*1e3:.3g}' + r'}',
                                    'b':lambda x: r'\num{'+ f'{x*1e3:.3g}' + r'}',
                                    'c':lambda x: r'\num{'+ f'{x*1e3:.3g}' + r'}',
                                    'volume': lambda x: r'\num{'+ f'{x*1e12:.3g}' + r'}',
                                    'density': lambda x: int(np.round(x,0)),
                                    'occurrence': lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                                    'CSF': lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                                    'sphericity': lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                                    'names': lambda x: x[:-1] + letters[mix_ind] + x[-1],
                                    },
                        column_format=r'lp{13mm}p{10mm}p{10mm}p{10mm}p{10mm}p{10mm}p{10mm}p{10mm}p{18mm}p{10mm}',
                        index=False,
                        label=f'tab:mp_mix_{mix_ind}',
                        caption=f'Microplastic mix {mix_ind}'

                     )
        for i in range(len(use_cols)):
            lat = lat.replace(use_cols_alias[i], col_names[i])

        file_path = 'D:\\table_test.tex'
        with open(file_path, 'a') as f:

            f.write(lat)

            f.write("\n\n\n")

def save_sample_table(file = f"D:\\server_files\\server_inputs\\sample_server_24.xlsx"):

    samples= pd.read_excel(file)
    #print(samples['names'])

    print(samples.columns)

    #mps = mps[['names', 'density', 'alow', 'a', 'aupp', 'b', 'c', 'CSF', 'sphericity', 'volume', 'occurrence']]
    use_cols = ['a7',
                'a8',
                'beta1',
                'beta2',
                'beta3',
                'beta4',
                'wwtps_rem_eff_primary',
                'wwtps_rem_eff_secondary',
                'wwtps_rem_eff_advanced',
                'wwtps_washing_mp',
                'wwtps_fiber_fraction',
                'wwtps_fragment_fraction',
                'wwtps_film_fraction',
                'wwtps_bead_fraction',
                'wwtps_foam_fraction'
                ]

    use_cols_alias = ['col1a7',
                'col2a8',
                '3beta1',
                '4beta2',
                '5beta3',
                '6beta4',
                '7wwtps_rem_eff_primary',
                '8wwtps_rem_eff_secondary',
                '9wwtps_rem_eff_advanced',
                '10wwtps_washing_mp',
                '11wwtps_fiber_fraction',
                '12wwtps_fragment_fraction',
                '13wwtps_film_fraction',
                '14wwtps_bead_fraction',
                '15wwtps_foam_fraction'
                ]
    col_names = [r'$\gamma_7$',
                 r'$\gamma_8$ $\times 10^{-6}$ \newline (\si{s^2 kg^{-1}})',
                 r'$\beta_1$',
                 r'$\beta_2$',
                 r'$\beta_3$',
                 r'$\beta_4$',
                 r'$r_{eff,P}$',
                 r'$r_{eff,S}$',
                 r'$r_{eff,A}$',
                 r'$N_{MP,wash}$',
                 r'$p_{MPfib}$',
                 r'$p_{MPfra}$',
                 r'$p_{MPfil}$',
                 r'$p_{MPb}$',
                 r'$p_{MPfoa}$',
                ]
    lat = samples.to_latex(
                    columns=use_cols,
                    header=use_cols_alias,
                    formatters={
                        'a7':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'a8':lambda x: r'\num{'+ f'{x*1e6:.3g}' + r'}',
                        'beta1':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'beta2':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'beta3':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'beta4':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_rem_eff_primary':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_rem_eff_secondary':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_rem_eff_advanced':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_washing_mp':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_fiber_fraction':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_fragment_fraction':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_film_fraction':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_bead_fraction':lambda x: r'\num{'+ f'{x:.3g}' + r'}',
                        'wwtps_foam_fraction':lambda x: r'\num{'+ f'{x:.3g}' + r'}'
                    },
                    index=True,
                    label='tab:full_samples',
                    column_format='rrp{8mm}p{8mm}p{10mm}p{8mm}p{8mm}p{8mm}'
                                  'p{8mm}p{8mm}p{13mm}p{8mm}p{8mm}p{8mm}p{10mm}p{8mm}',
                    caption=r'Full samples as generated from the ranges presented in table \ref{tab:uncertainties}'

                 )
    for i in range(len(use_cols)):
        lat = lat.replace(use_cols_alias[i], col_names[i])

    file_path = 'D:\\sample_table.tex'
    with open(file_path, 'w') as f:

        f.write(lat)
def plot_retention_rates(show = False):
    names = ["small_09", "med_09", "big_09", "small_11", "med_11", "big_11", "small_12", "med_12", "big_12"]
    col_names = [ "0.15 mm < r < 0.25 mm", "0.45 mm < r < 0.55 mm", "1 mm < r < 2 mm"]
    row_names = [r"$\rho$ = 0.9", r"$\rho$ = 1.1 kg/m$^3$", r"$\rho$ = 1.2"]
    retention_fractions = {}
    files = ["D:\\test_run_new_inputs.nc", "D:\\test_run_new_mass_balance.nc", "D:\\test_run_new_mb_dis_first.nc"]
    data_titles = ["Without MB", "Emissions first", "Discharge first"]
    neg_vals = {}
    for f in range(len(files)):
        data = nc.Dataset(files[f], "r")
        for switch in range(2):
            if switch == 0:
                plot_log = True
            else:
                plot_log = False
            ncol = 3
            nrow = 3
            fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(10,10), sharey="row")

            for i in range(len(names)):
                y_ind = np.floor(i/ncol).astype(int)
                x_ind = i%ncol
                ax_ind = (y_ind,x_ind)
                temp = np.ma.getdata(data[f"sed_mp_{names[i]}"][:,:,:])/data[f"sus_mp_{names[i]}"][:,:,:]
                retention_fractions[names[i]] = []
                for j in range(len(data["time"])):
                    retention_fractions[names[i]].append(temp[j, temp[j].mask == False])
                labels = num2date(data["time"][:].astype(int), units=data['time'].units)
                for k in range(len(labels)):
                    if k == 0:
                        labels[k] = labels[k].strftime("%Y-%m-%d")
                    else:
                        labels[k] = labels[k].strftime("%m-%d")
                axs[ax_ind].boxplot(retention_fractions[names[i]])
                axs[ax_ind].set_xticks(range(1, len(labels) + 1), labels, rotation=90)
                #axs[ax_ind].set_title(names[i])
                if y_ind > 0 and plot_log:
                    axs[ax_ind].set_yscale("log")
                if y_ind == nrow-1:
                    axs[ax_ind].set_xlabel(col_names[i%ncol])
                if x_ind == 0:
                    axs[ax_ind].set_ylabel(row_names[np.floor(i/ncol).astype(int)])
            if plot_log:
                with open(f'D:\\test_ret_fra_{data_titles[f]}.pkl', 'wb') as fi:
                    pickle.dump(retention_fractions, fi)

            # with open('saved_dictionary.pkl', 'rb') as f:
            #     loaded_dict = pickle.load(f)
            if plot_log:
                text= "_log"
            else:
                text = ""

            plt.tight_layout()
            plt.savefig(f"D:\\test_ret_fra_{data_titles[f]}{text}.png", dpi=150)
            if show:
                plt.show()
            else:
                plt.close()
            print(f"done with plot {data_titles[f]} number {switch}")
    return

def plot_retention_percent_subplots(show = False, plot_log = False, names = None,
                           col_names = None, row_names = None, files = None, data_titles = None):
    if not files:
        files = ["D:\\test_run_new_mb_dis_first.nc"]
        data_titles = ["Discharge first"]
    if not names:
        names = ["small_09", "med_09", "big_09", "small_11", "med_11", "big_11", "small_12", "med_12", "big_12"]
        col_names = [ "0.15 mm < r < 0.25 mm", "0.45 mm < r < 0.55 mm", "1 mm < r < 2 mm"]
        row_names = [r"$\rho$ = 0.9", r"$\rho$ = 1.1 kg/m$^3$", r"$\rho$ = 1.2"]
    retention_fractions = {}
    neg_vals = {}
    for f in range(len(files)):
        data = nc.Dataset(files[f], "r")

        ncol = 3
        nrow = 3
        fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(10,14), sharey="row")

        for i in range(len(names)):
            y_ind = np.floor(i/ncol).astype(int)
            x_ind = i%ncol
            ax_ind = (y_ind,x_ind)
            temp = np.ma.getdata(data[f"sed_mp_{names[i]}"][:,:,:])/(data[f"sus_mp_{names[i]}"][:,:,:] +
                                                                     np.ma.getdata(
                                                                         data[f"sed_mp_{names[i]}"][:,:,:]
                                                                     )
                                                                     )
            retention_fractions[names[i]] = []
            for j in range(len(data["time"])):
                retention_fractions[names[i]].append(temp[j, temp[j].mask == False])
            labels = num2date(data["time"][:].astype(int), units=data['time'].units)
            for k in range(len(labels)):
                if k == 0:
                    labels[k] = labels[k].strftime("%Y-%m-%d")
                else:
                    labels[k] = labels[k].strftime("%m-%d")
            axs[ax_ind].boxplot(retention_fractions[names[i]])
            axs[ax_ind].set_xticks(range(1, len(labels) + 1), labels, rotation=90)
            #axs[ax_ind].set_title(names[i])
            if y_ind > 0 and plot_log:
                axs[ax_ind].set_yscale("log")
            if y_ind == nrow-1:
                axs[ax_ind].set_xlabel(col_names[i%ncol])
            if x_ind == 0:
                axs[ax_ind].set_ylabel(row_names[np.floor(i/ncol).astype(int)])

        with open(f'D:\\test_ret_percent_{data_titles[f]}.pkl', 'wb') as fi:
            pickle.dump(retention_fractions, fi)

        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        if plot_log:
            text= "_log"
        else:
            text = ""

        plt.tight_layout()
        plt.savefig(f"D:\\test_ret_percent_{data_titles[f]}{text}.png", dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"done with plot {data_titles[f]}")
    return


def plot_retention_percent(show=False, plot_log=False, names=None, files=None, data_titles=None, calc=True):
    if not files:
        files = ["D:\\test_run_new_mb_dis_first.nc"]
        data_titles = ["Discharge first"]
    if not names:
        names = ["small_09", "med_09", "big_09", "small_11", "med_11", "big_11", "small_12", "med_12", "big_12"]
        col_names = ["0.15 mm < r < 0.25 mm", "0.45 mm < r < 0.55 mm", "1 mm < r < 2 mm"]
        row_names = [r"$\rho$ = 0.9", r"$\rho$ = 1.1 kg/m$^3$", r"$\rho$ = 1.2"]
    retention_fractions = {}
    neg_vals = {}
    for f in range(len(files)):
        data = nc.Dataset(files[f], "r")

        fig = plt.figure(figsize=(20, 10))

        for i in range(len(names)):
            if calc:
                temp = np.ma.getdata(data[f"sed_mp_{names[i]}"][:, :, :]) / (data[f"sus_mp_{names[i]}"][:, :, :] +
                                                                             np.ma.getdata(
                                                                                 data[f"sed_mp_{names[i]}"][:, :, :]
                                                                             )
                                                                             )
                retention_fractions[names[i]] = []

                for j in range(len(data["time"])):
                    retention_fractions[names[i]].append(temp[j, temp[j].mask == False])
            labels = num2date(data["time"][:].astype(int), units=data['time'].units)
            for k in range(len(labels)):
                if k == 0:
                    labels[k] = labels[k].strftime("%Y-%m-%d")
                else:
                    labels[k] = labels[k].strftime("%m-%d")
        if not calc:
            with open(f'D:\\test_ret_percent_{data_titles[f]}.pkl', 'rb') as fi:
                retention_fractions = pickle.load(fi)

        plt_data = []
        for timestep in range(len(data["time"])):
            plt_data.append([retention_fractions[var][timestep].flatten() for var in retention_fractions.keys()])

        for i in range(len(plt_data)):
            boxplt = plt.boxplot(plt_data[i][:], positions=[3*i + j for j in [-0.6,-0.2,0.2,0.6]], patch_artist=True,
                                 widths=0.3)
            colors = ["indianred", "salmon", "royalblue", "cornflowerblue"]
            for patch in range(len(boxplt['boxes'])):
                boxplt["boxes"][patch].set_facecolor(colors[patch % 4])
                boxplt["medians"][patch].set_color('black')
        plt.title(data_titles[f])
        plt.xticks(3*np.array([*range(len(labels))]), labels, rotation=90)
            # axs[ax_ind].set_title(names[i])


        if plot_log:
            plt.yscale("log")
        plt.xlabel("Time (days)")
        plt.ylabel("Retention percent")
        if calc:
            with open(f'D:\\test_ret_percent_{data_titles[f]}.pkl', 'wb') as fi:
                pickle.dump(retention_fractions, fi)

        custom_lines = [Patch(facecolor=colors[i], edgecolor='black',
                         label=names[i]) for i in range(4)]

        plt.legend(handles=custom_lines,bbox_to_anchor=(1.01,1), loc='upper left')
        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        if plot_log:
            text = "_log"
        else:
            text = ""

        plt.tight_layout()
        plt.savefig(f"D:\\test_ret_percent_{data_titles[f]}{text}.png", dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"done with plot {data_titles[f]}")
    return

def create_time_labels(data):
    labels = num2date(data["time"][:].astype(int), units=data['time'].units)
    for k in range(len(labels)):
        if k == 0:
            labels[k] = labels[k].strftime("%Y-%m-%d")
        else:
            labels[k] = labels[k].strftime("%m-%d")
    return labels


def boxplot_time_multiple_cats(mp_cats, time_labels, axs, pars=None,
                               colors=None, plot_log=False, legend=True, lower_bd = 0.1, markersize = 3):

    """Where we assume that mp_cats is a dictionary with cat: [t x data] format."""

    if pars is None:
        pars = [3, 0.6, 0.75]
    if colors is None:
        colors = ["indianred", "salmon", "royalblue", "cornflowerblue"]
    plt_data = []
    for timestep in range(len(time_labels)):
        plt_data.append([mp_cats[var][timestep,
                                      np.where(mp_cats[var][timestep] > lower_bd)
                                     ].flatten() for var in mp_cats.keys()])

    for i in range(len(plt_data)):
        pos = [-pars[1] + k * pars[1]/(len(mp_cats)-1) for k in range(len(mp_cats))]
        boxplt = axs.boxplot(plt_data[i][:],
                             positions=[pars[0] * i + j for j in pos], patch_artist=True,
                             widths=pars[2] * 2 * pars[1] / ((len(mp_cats) - 1)),
                             flierprops={'markersize':markersize}

                             )

        for patch in range(len(boxplt['boxes'])):
            boxplt["boxes"][patch].set_facecolor(colors[patch % len(mp_cats)])
            boxplt["medians"][patch].set_color('black')

    axs.set_xticks(pars[0] * np.array([*range(len(time_labels))]), time_labels, rotation=90)
    # axs[ax_ind].set_title(names[i])
    if legend:
        custom_lines = [Patch(facecolor=colors[i], edgecolor='black',
                              label=list(mp_cats.keys())[i]) for i in range(len(mp_cats.keys()))]

        axs.legend(handles=custom_lines, bbox_to_anchor=(1.01, 1), loc='upper left')
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)


def get_stocks_per_type(names, file, stock_names, flow_arr, sort_by=None, time_len=None):
    if sort_by is None:
        sort_by = 'mp'
    data = nc.Dataset(file, "r")
    if time_len is None:
        time_len = len(data['time'])
    labels = create_time_labels(data)
    final_stocks = {}
    if sort_by == 'stock':
        for s in range(len(stock_names)):
            final_stocks[stock_names[s]] = {}
    for n in range(len(names)):
        mp_stocks = {}
        for s in range(len(stock_names)):
            if stock_names[s] == 'suspended':
                arr = data[f"sus_mp_{names[n]}"][:time_len,:,:]
                print(arr.shape)
                where = np.where(flow_arr != 5)
                print(arr[:,where[0], where[1]].shape)
                mp_stocks[stock_names[s]] = arr[:time_len,where[0], where[1]]
            elif stock_names[s] == 'sediment':

                arr = data[f"sed_mp_{names[n]}"][:time_len,:,:]
                where = np.where(flow_arr != 5)
                mp_stocks[stock_names[s]] = arr[:time_len, where[0], where[1]]
            elif stock_names[s] == 'ocean':
                where = np.where(flow_arr == 5)
                dat1 = data[f"sus_mp_{names[n]}"][:time_len,:,:]
                dat2 = data[f"sed_mp_{names[n]}"][:time_len,:,:]
                mp_stocks[stock_names[s]] = np.ma.getdata(dat1[:time_len,where[0], where[1]]) + \
                                             np.ma.getdata(dat2[:time_len, where[0], where[1]])
        if sort_by == 'mp':
            final_stocks[names[n]] = mp_stocks
        elif sort_by == 'stock':
            for s in range(len(stock_names)):
                final_stocks[stock_names[s]][names[n]] = mp_stocks[stock_names[s]]
    return final_stocks, labels


def plot_stocks_over_time(plt_stocks, stock_names, labels, ax, plt_type='boxplot', markers=None, colors=None, legend=True,
                          log_y=True, normalize=False):

    if colors is None:
        colors = ["indianred", "salmon", "royalblue", "cornflowerblue"]

    if 'boxplot' in plt_type:
        boxplot_time_multiple_cats(plt_stocks, labels, ax, colors=colors, pars=[3,1,0.3], legend=legend)

    elif 'scatter' in plt_type:
        if markers is None:
            markers = ['o', '^', 's']

        for s in range(len(stock_names)):
            ax.plot(
                  [*range(len(labels))],
                  np.sum(plt_stocks[stock_names[s]], axis=1),
                  linewidth = 0,
                  marker=markers[s],
                  label=stock_names[s],
                  color=colors[s]
                )

        if legend:
            ax.legend()
    elif 'bar' in plt_type:
        if normalize:
            normalizations = np.zeros(len(labels))
            for s in range(len(stock_names)):
                normalizations += np.sum(plt_stocks[stock_names[s]], axis=1)
        bottom = np.zeros(len(labels))
        for s in range(len(stock_names)):

            new_dat = np.sum(plt_stocks[stock_names[s]], axis=1)
            if normalize:
                new_dat = new_dat/normalizations
            p = ax.bar(labels, new_dat, 0.5,
                       label=stock_names[s],
                       color=colors[s],
                       bottom=bottom
                )
            bottom += new_dat
            #print(bottom)

        if legend:
            ax.legend()
    if log_y:
        ax.set_yscale('log')
    return


def create_subplot_figure(num_plots, figsize=(12,10), sharey='all'):
    num_cols = int(np.ceil(np.sqrt(num_plots)))  # Calculate number of columns
    num_rows = int(np.ceil(num_plots / num_cols))  # Calculate number of rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, sharey=sharey)  # Adjust figsize as needed

    # # Flatten the axs array if it's not already flat
    # if not isinstance(axs, (list, np.ndarray)): #I don't know if this ever happens and if we need this check
    #     axs = [axs]

    return fig, axs



def filter_ocean_and_river_stocks(raster, raster_keys, flow_map, t_step):
    out_dict = {}
    for i in range(len(raster_keys)):

        if 'ocean' in raster_keys[i]:
            key = raster_keys[i][len('ocean_mp_'):]
            temp = raster['sus_mp_' + key][t_step, :, :] + \
                   raster['sed_mp_' + key][t_step, :, :]

            temp = np.where(flow_map == 5, temp, 0)
        else:
            temp = raster[raster_keys[i]][t_step, :, :]
            temp = np.where(flow_map != 5, temp, 0)
        out_dict[raster_keys[i]] = temp
    return out_dict


def get_data_per_raster(areas, rasters, flow_map, mp_keys, num_plastics=15,num_basins=53, t_step=-1):
    world_mask = areas["mask5minFromTop_map"][:, :].astype(int)
    data_arr = np.zeros((num_plastics, num_basins))
    data_dict = filter_ocean_and_river_stocks(rasters, mp_keys, flow_map, t_step)
    for i in range(num_basins):
        for j in range(len(mp_keys)):
            temp = data_dict[mp_keys[j]]
            stat = np.nansum(temp[np.where(world_mask == i + 1)])
            data_arr[j,i] = stat
    return data_arr


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
def get_land_areas(areas, num_basins=53):
    areas = areas["mask5minFromTop_map"][:, :].astype(int)
    lengths = np.zeros(num_basins)
    for i in range(num_basins):
        lengths[i] = np.sum(np.where(areas == i+1, 1,0))

    return lengths

def heatmap_parameters():
    directions = nc.Dataset('D:\\inputs\\channel_parameters_extended.nc')["lddMap"][:,:]

    land = nc.Dataset('D:\\world_basins\\pcr_basins\\mask5minFromTop.nc')
    n=''
    mp_keys = [
        f"fiber{n}1",
        f"fiber{n}2",
        f'fiber{n}3',
        f'fragment{n}1',
        f'fragment{n}2',
        f'fragment{n}3',
        f'bead{n}1',
        f'bead{n}2',
        f'bead{n}3',
        f'foam{n}1',
        f'foam{n}2',
        f'foam{n}3',
        f'film{n}1',
        f'film{n}2',
        f'film{n}3'
    ]
    keys_to_get = [f"sed_mp_{mp_keys[i]}" for i in range(len(mp_keys))]
    #   total_dat = np.zeros((20,15,53))
    # for i in range(5):
    # output = nc.Dataset(f'D:\\out_tests\\final_local0.nc')
    # data = get_data_per_raster(land,output,directions,keys_to_get)
    # print(data.shape)
    # #     total_dat[i] = data
    # np.save('D:\\arr_test', total_dat)
    data = np.load('D:\\arr_test.npy')
    sizes = get_land_areas(land)
    # print(sizes)
    # print( np.maximum(sizes/np.max(sizes), 0.1))
    # hey = np.cumsum(sizes)
    # print(hey)
    # print(len(hey))
    coeff_of_variation = np.std(data, axis=0)/np.mean(data, axis=0)
    coeff_of_variation[np.where(coeff_of_variation == np.nan)] = 0
    create_heatmap(coeff_of_variation, plastic_names=mp_keys, land_areas=sizes, plot_log=False)


def create_global_time_plots():
    # Example usage:
    # Now you can access each subplot using axs[row_index, col_index]
    # For example, axs[0, 0] represents the subplot in the first row and first column.
    # You can loop through axs to plot on each subplot.

    # show=True
    # plot_log=False
    # names=['sph_lar', 'sph_sma', 'fib_lar', 'fib_sma']
    # files=['D:\\test_yu_4cat_1.nc', 'D:\\test_nizetto_4cat_1.nc']
    # data_titles=['Yu', "Nizetto"]
    # calc=True
    # plot_retention_percent(show=show, plot_log=plot_log, names=names, files=files, data_titles=data_titles, calc=calc)
    names=[
        "fiber",
        "fiber_large",
        "plate_thin",
        "plate",
        "disk_thin",
        "disk_thick",
        "cylinder",
        "prism",
    ]
    colors = [plt.cm.Paired(i) for i in range(len(names))]


    print(len(colors))
    print(colors)
    file='D:\\test_yu_8cat_1000.nc'
    stocks = ['suspended', 'sediment','ocean']
    flow_arr = nc.Dataset('D:\\inputs\\channel_parameters_extended.nc', 'r')['lddMap'][:,:]
    plt_stocks, labels = get_stocks_per_type(names,file,stocks,flow_arr,sort_by='stock')
    # with open("D:\\out_tests\\test_yu_mp_100.pkl", 'wb') as f:
    #     pickle.dump([plt_stocks, labels], f)
    #
    # with open('D:\\out_tests\\test_yu_mp_100.pkl', 'rb') as f:
    #     inputs = pickle.load(f)

    inputs = [plt_stocks, labels]
    ind = [0,1,2]
    #ind = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
    left, top = 0.02, 0.95
    #fig, ax = create_subplot_figure(len(ind), sharey=False)
    fig, ax = plt.subplots(3, sharey=True, sharex=True, figsize=(10,18))
    plt_log = False
    for i in range(len(stocks)):
        ax[ind[i]].yaxis.grid(True)
        ax[ind[i]].xaxis.grid(True)

        plot_stocks_over_time(inputs[0][stocks[i]],names,inputs[1],ax[ind[i]], log_y=plt_log, plt_type='bar',
                              legend=i == 2,
                              colors=colors,
                              normalize=False
                              )
        if i == len(stocks) - 1:
            ax[ind[i]].tick_params(axis='x', labelrotation=90)
            #ax[ind[i]].set_xticks(ax[ind[i]].get_xticklabels(),rotation=90)
        ax[ind[i]].text(left, top, names[i],
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax[ind[i]].transAxes)

    plt.tight_layout()
    plt.savefig('D:\\out_tests\\yu_stocks_1000_mps_bar.png', dpi=160)
    plt.show()


    #PLOT BY MP
    # ind = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
    # left, top = 0.02, 0.95
    # fig, ax = create_subplot_figure(len(names))
    # plt_log = True
    # for i in range(len(names)):
    #     plot_stocks_over_time(inputs[0][names[i]],stocks,inputs[1],ax[ind[i]], log_y=plt_log, boxplot=False,
    #                           legend= i==2
    #                           )
    #     ax[ind[i]].text(left, top, names[i],
    #                     horizontalalignment='left',
    #                     verticalalignment='top',
    #                     transform=ax[ind[i]].transAxes)
    #     ax[ind[i]].yaxis.grid(True)
    #     ax[ind[i]].xaxis.grid(True)
    #
    # plt.tight_layout()
    # plt.savefig('D:\\out_tests\\yu_stocks_t_total_500_log.png', dpi=160)
    # plt.show()
def create_heatmap(data, n_y=15, n_sinks=53, y_names=None, num_divs=10, land_areas=None, plot_log=True,
                   div_dist=3, cmap='Blues', hlines=True, fig=None, ax=None, finish_plot=False):
    if y_names is None:
        y_names = [f'plastic_{i}' for i in range(n_y)]
    if plot_log:
        print(np.nanmax(data))
        print(max(np.nanmin(data[np.nonzero(data)]), 0.001))
        norm = LogNorm(
            vmin=np.nanmin(data[np.nonzero(data)]),
            vmax=np.nanmax(data))
    else:
        norm = None
    if fig is None:
        fig, ax = plt.subplots(figsize=(22,6))


    country_dividers = np.array([0,9,23,25,34,46,49])
    label_locs = (country_dividers + np.roll(country_dividers, -1))/2 -0.5
    label_locs[-1] += n_sinks/2
    print(np.min(data[np.nonzero(data)]))
    print(np.max(data[np.nonzero(data)]))

    if land_areas is None:
        y_tick_offset = 0
        da = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)

        ax.vlines(country_dividers - 0.5, ax.get_ylim()[0], ax.get_ylim()[1], colors='black')
        if hlines:
            ax.hlines(np.array([div_dist * (i + 1) - 0.5 for i in range(int(n_y / div_dist))]), ax.get_xlim()[0],
                      ax.get_xlim()[1], colors='black')
    else:
        y_tick_offset = 0.5
        y_ax = np.array([*range(n_y+1)])
        area_sizes = np.cumsum(np.maximum(land_areas/np.max(land_areas), 0.15))
        basin_ax = np.zeros(len(area_sizes)+1)
        basin_ax[1:] = area_sizes
        x, y = np.meshgrid(basin_ax, y_ax)
        da = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm)
        ax.vlines(basin_ax, ax.get_ylim()[0], ax.get_ylim()[1], colors='black', linewidths=0.7)
        ax.vlines(basin_ax[country_dividers], ax.get_ylim()[0], ax.get_ylim()[1], colors='black', linewidths=3.5)
        if hlines:
            ax.hlines(y_ax[::3], ax.get_xlim()[0], ax.get_xlim()[1], colors='black')

    ax.figure.colorbar(da, pad=0.01)
    ax.set_facecolor('lightgrey')
    # Setting all the labels
    hoi = ax.twiny()
    if land_areas is None:
        ax.set_xticks(label_locs)
        hoi.set_xticks([4 + 5*i for i in range(num_divs)], labels=[5* (i+1) for i in range(num_divs)], fontsize=16)
    else:
        xtick_locs = (basin_ax[country_dividers] + np.roll(basin_ax[country_dividers], -1))/2
        xtick_locs[-1] += basin_ax[-1]/2
        ax.set_xticks(xtick_locs)
        arr = np.array([5 + 5 * i for i in range(num_divs)])
        hoi.set_xticks((basin_ax[arr] + basin_ax[arr - 1])/2,
                       labels=[5 * (i + 1) for i in range(num_divs)], fontsize=16)

        # Drawing horizontal and vertical dividers

    ax.set_yticks(np.array([*range(n_y)])+y_tick_offset, labels=y_names, fontsize=16)
    ax.set_xticklabels(['Africa & \nMiddle East','Asia','  Central \n America',
                        'Europe\n & NI','North America',
                        'Oceania','South \n America'], fontsize=16, ha='center', rotation=90)
    hoi.set_xlim(ax.get_xlim())
    if finish_plot:
        plt.tight_layout()
        plt.savefig('D:\\out_tests\\heatmap', dpi=300)
        plt.show()

def create_heatmap_subplots(data,
                            dt=False, time_plot=False, filename='D:\\out_tests\\heatmap_test', tstep=-1,
                            plot_log=True,sum_vals=True
                            ):
    #data = np.load('D:\\full_data_test.npy')
    land = nc.Dataset('D:\\world_basins\\pcr_basins\\mask5minFromTop.nc')
    plastic_names = nc.Dataset("D:\\out_tests\\final_local5_0.nc").mp_names.copy()
    plot_names = ['Suspended', "Sediment", "Sinks", "Total"]
    cmaps = ['Blues', 'YlOrBr', "Greens", "Purples"]
    sizes = get_land_areas(land)
    fig, ax = plt.subplots(ncols=2,nrows=2, figsize=(28, 18))


    inds = [(0,0), (0,1), (1,0), (1,1)]
    if time_plot:
        sum_ax = 1
        y_names = [*range(1, 21)]
        y_num = 20
        plot_hlines = False
    else:
        sum_ax = 0
        y_names = plastic_names
        y_num = 15
        plot_hlines = True

    for i in range(4):
        if time_plot and sum_vals:
            temp_dat = np.nansum(data[:,i,:,:], axis=sum_ax)

            if dt:
                temp_dat[1:] -= temp_dat[:-1]
                plot_log = False
        elif not time_plot:
            temp_dat = data[tstep,i,:,:]
        else:
            temp_dat = data[:,i,:]
        print(temp_dat.shape)
        create_heatmap(temp_dat, n_y=y_num, y_names=y_names, fig=fig, ax=ax[inds[i]],
                       finish_plot=False, land_areas=sizes, cmap=cmaps[i], hlines=plot_hlines, plot_log=plot_log)



        ax[inds[i]].set_title(plot_names[i])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def uncertainty_scatter_plot(total_data, uncertainties, uncert_names,
                             percentage = False, plot_log=True, concat_data=False,
                             out_file='D:\\final_outputs\\uncertainty_scatter', n_mix=8, small_plot=False,
                             legend=False):
    if n_mix != 7:
        mix_inds = [*range(n_mix)]
    else:
        mix_inds = [0, 1, 2, 3, 4, 5, 7]

    if concat_data:
        total_data = np.zeros((24 * n_mix, 20, 4, 15, 53))
        for run_ind in range(n_mix):
            for i in range(24):
                data = np.load(f"D:\\final_outputs\\agg_res\\mix_{mix_inds[run_ind]}\\agg_res_{mix_inds[run_ind]}_{i}.npy")
                total_data[run_ind * 24 + i, :, 0:3, :, :] = data[:20]
        total_data[:, :, 3, :, :] = np.sum(total_data[:, :, :3, :, :], axis=2)
        np.save("D:\\final_outputs\\total_agg_1.npy", total_data)
    if not small_plot:
        num_rows = 3
        num_cols = 5
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(28,18))
    else:
        num_cols = 3
        num_rows = 2
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(21,12))
    markers= ['+', 'x', '*','d', 's', '.', '<', '>']
    colors = ['darkblue', 'orange', 'green', 'purple']
    stocks = ['Suspended', 'Sedimented', 'Exported', "Total"]
    fig_ind = 0
    print(total_data.shape)

    if percentage == True:
        div = np.nansum(total_data[:,:,-1,:,:], axis=(-1,-2))
        total_data = total_data/div[:,:,None,None,None]
    for i in range(num_rows):
        for j in range(num_cols):
            for m in range(n_mix):
                for d in range(4):
                    temp =np.nansum(total_data[m * 24:(m + 1) * 24, -1, d, :, :], axis=(-1, -2))
                    ax[i, j].scatter(uncertainties[uncert_names[fig_ind]],
                                     temp,
                                     c=colors[d],
                                     marker=markers[m])

            ax[i, j].set_xlabel(uncert_names[fig_ind], fontsize=14)
            if j == 0:
                ax[i, j].set_ylabel('Final microplastic emissions', fontsize=14)
            if plot_log:
                ax[i,j].set_yscale('log')
            fig_ind += 1
    if legend:
        handles = [
            Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='darkgrey', markeredgecolor='darkgrey',
                   label= f'Mix {i+1}',
                   markersize=11, linewidth=2)
            for i in range(n_mix)
        ]
        for i in range(4):
            handles.append(Patch(color=colors[len(stocks)-1-i], label=stocks[len(stocks)-1-i]))
        plt.figlegend(handles=handles, bbox_to_anchor=(0.98,0.6))
    #plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()


def create_heatmap_relative():
    total_data = np.load("D:\\final_outputs\\total_agg_1.npy")
    #dat = np.std(total_data, axis=0)/np.mean(total_data, axis=0)
    dat = np.nansum(total_data, axis=-2)
    print(dat.shape)
    dat = dat[:,:,2,:]/dat[:,:,1,:]
    dat[np.where(dat == np.inf)] = np.nan
    dat = np.nanmean(dat, axis=0)
    print(dat)
    #dat = np.std(dat, axis=0)/np.mean(dat, axis=0)
    #create_heatmap_subplots(data = dat, dt=False, time_plot=True, plot_log=False, sum_vals=False,
    #                        filename='D:\\final_outputs\\std_time_series')
    i = 0
    land = nc.Dataset('D:\\world_basins\\pcr_basins\\mask5minFromTop.nc')
    plastic_names = nc.Dataset("D:\\out_tests\\final_local5_0.nc").mp_names.copy()
    #plot_names = ['Suspended', "Sediment", "Sinks", "Total"]
    cmaps = ['RdYlBu']
    sizes = get_land_areas(land)
    fig, ax = plt.subplots(figsize=(14, 10))
    time_plot = True
    plot_log = True
    #inds = [(0,0), (0,1), (1,0), (1,1)]
    if time_plot:
        sum_ax = 1
        y_names = [*range(1, 21)]
        y_num = 20
        plot_hlines = False
    else:
        sum_ax = 0
        y_names = plastic_names
        y_num = 15
        plot_hlines = True
    create_heatmap(dat, n_y=y_num, y_names=y_names, fig=fig, ax=ax,
                   finish_plot=False, land_areas=sizes, cmap=cmaps[i], hlines=plot_hlines, plot_log=plot_log)



    ax.set_title('mean sediment sink fraction')
    plt.tight_layout()
    plt.savefig('D:\\final_outputs\\sediment_sink_fraction', dpi=300)
    plt.show()
    # rng = np.random.default_rng(123456)
    # total_data = rng.random((48,20,4,20,53))
    # for i in range(1,4):
    #     total_data[:,:,i,:,:] *= i *1.5

    #print(uncertainties['beta1'])
def create_pair_plot(save_data = False, n_mix=8):
    if save_data:
        total_data = np.load("D:\\final_outputs\\total_agg_1.npy")
        pair_df = pd.DataFrame({"Suspended": np.sum(total_data[:, -1, 0, :, :], axis=(-1, -2)),
                                "Sedimented": np.sum(total_data[:, -1, 1, :, :], axis=(-1, -2)),
                                "Exported": np.sum(total_data[:, -1, 2, :, :], axis=(-1, -2)),
                                "Total": np.sum(total_data[:, -1, 3, :, :], axis=(-1, -2)),
                                "Mix": np.array([int(i / 24) for i in range(24*n_mix)])
                                })

        pair_df.to_excel("D:\\final_outputs\\pair_plot_df.xlsx")

    pair_df = pd.read_excel("D:\\final_outputs\\pair_plot_df.xlsx")
    g = sns.PairGrid(pair_df[["Suspended", "Sedimented", "Exported", "Total", "Mix"]], diag_sharey=False,
                     hue='Mix', palette='tab10')
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot)
    g.map_upper(sns.kdeplot)
    plt.savefig("D:\\final_outputs\\pair_plot.png", dpi=150)
    plt.show()

def total_time_plot(per_mix=True, n_mix=8, plt_type='normal'):
    total_data = np.load("D:\\final_outputs\\total_agg_1.npy")
    print(total_data.shape)
    cmap = plt.get_cmap("tab10")
    markers= ['+', 'x', '*','d', 's', '.', 5, '>']
    dat = np.sum(total_data, axis=(-1,-2))

    print(dat.shape)
    if plt_type == 'small':
        fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(12,15))
    elif type== 'large':
        fig, ax = plt.subplots(ncols=n_mix, nrows=4, figsize=(30,16), sharey='row')
    else:
        fig, ax = plt.subplots(ncols=4, figsize=(20, 6))
    inds = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i in range(n_mix):
        if per_mix and plt_type =='normal':

            for j in range(4):
                # ax[j].plot([*range(1,21)], means[:,j].T, color=colors[i])
                # ax[j].fill_between([*range(1,21)], np.transpose(means[:,j]+stds[:,j]), np.transpose(means[:,j]-stds[:,j]),
                #                    color=colors[i],
                #                    alpha=0.5)
                ax[j].plot([*range(1,21)],dat[24*i:24*(i+1),:,j].T, color=cmap(i), linewidth=0.7)
        elif plt_type =='large':
            means = np.mean(dat[24 * i:24 * (i + 1)], axis=0)

            upper_quantiles = np.percentile(dat[24 * i:24 * (i + 1)], 75,axis=0)
            lower_quantiles = np.percentile(dat[24 * i:24 * (i + 1)], 25,axis=0)
            for j in range(4):
                ax[j,i].plot([*range(1,21)], means[:,j].T, color=cmap(i))
                ax[j,i].fill_between([*range(1,21)], np.transpose(upper_quantiles[:,j]),
                                   np.transpose(lower_quantiles[:,j]),
                                   color=cmap(i),
                                   alpha=0.5)
                #ax[j, i].set_yscale('log')
                if j == 0:
                    ax[j,i].set_title(f'Mix {i}')
        elif plt_type == 'small':
            means = np.mean(dat[24 * i:24 * (i + 1)], axis=0)

            q3 = np.percentile(dat[24 * i:24 * (i + 1)], 75, axis=0)
            q1 = np.percentile(dat[24 * i:24 * (i + 1)], 25, axis=0)
            stds = np.std(dat[24 * i:24 * (i + 1)], axis=0)
            styles = ['dashed', '-']
            names = ['Suspended', 'Sedimented']
            for j in range(3):
                for k in range(j):
                    ax[j, 0].plot([*range(1, 21)], means[:, k].T, color='lightgrey', linestyle=styles[k], zorder=-1)

                    # ax[j, 1].plot([*range(1, 21)], np.transpose(stds[:, k] / means[:, k]), color='lightgrey',


                    #               linestyle='dashed')
                ax[j, 0].plot([*range(1, 21)], means[:, j].T, color=cmap(i))
                #ax[j, 1].plot([*range(1, 21)], np.transpose(stds[:, j]/means[:,j]), color=cmap(i))
                ax[j, 1].plot([*range(1, 21)], np.transpose((q3[:, j] - q1[:, j]) / (q3[:, j] + q1[:, j])),
                              color=cmap(i))
                #ax[j, 1].set_ylim([0.33, 0.52])
                ax[j, 1].set_ylim([0.2, 0.4])

                ax[j,0].ticklabel_format(axis='y', style='sci', useMathText=True)
                if j != 2:
                    ax[j, 0].set_xticks([])
                    ax[j, 1].set_xticks([])
                else:
                    ax[j, 0].set_xticks([*range(1, 21, 4)], [1996 + i for i in range(5)])
                    ax[j, 1].set_xticks([*range(1, 21, 4)], [1996 + i for i in range(5)])
                # ax[j, 1].fill_between([*range(1, 21)], np.transpose(upper_quantiles[:, j]),
                #                       np.transpose(lower_quantiles[:, j]),
                #                       color=cmap(i),
                #                       alpha=0.5)
                # ax[j, i].set_yscale('log')
                # if j == 0:
                #     ax[j,0].set_title('means')
                #     ax[j,1].set_title('variances')
            handles = [Line2D([0], [0], color=cmap(i), lw=4, label = f'Mix {i + 1}') for i in range(8)]
            ax[2,1].legend(handles=handles)
            for i in range(2):
                handles.append(Line2D([0], [0], color='lightgrey', lw=4, label = names[i], linestyle=styles[i]))
            ax[2,0].legend(handles=handles, handlelength=3)

        else:
            means=np.mean(dat, axis=0)
            stds = np.mean(dat, axis=0)
            ax[0].plot([*range(1, 21)], means[:,i].T)
            ax[0].fill_between([*range(1, 21)], np.transpose(means[:, i] + stds[:, i]),
                               np.transpose(means[:, i] - stds[:, i]),
                               alpha=0.5)
            #ax[j].plot([*range(1,21)],dat[:,:,j].T, color=colors[i], linewidth=0.7)


        if i == 3 and per_mix and plt_type=='normal':
            custom_lines = [Line2D([0], [0], color=cmap(i), lw=4, label=f"Mix {i}") for i in range(n_mix)]
            ax[i].legend(handles=custom_lines, bbox_to_anchor=(1.05, 0.95))
    #ax[0].set_yscale('log')
    if plt_type=='normal':
        ax[0].set_title("Suspended")
        ax[1].set_title("Sediment")
        ax[2].set_title("Exported")
        ax[3].set_title("Total")
    elif plt_type == 'small':
        units = [' (Number of particles)', r' ($\sigma / \mu$)']
        for i  in range(2):
            ax[0, i].set_ylabel("Suspended" + units[i])
            ax[1, i].set_ylabel("Sediment" + units[i])
            ax[2, i].set_ylabel("Exported" + units[i])
    else:
        ax[0,0].set_ylabel("Suspended")
        ax[1,0].set_ylabel("Sediment")
        ax[2,0].set_ylabel("Exported")
        #ax[3,0].set_ylabel("Total")
    plt.tight_layout()
    if plt_type == 'small':
        from matplotlib.transforms import Bbox
        cut_line = 5.9
        plt.savefig('D:\\final_outputs\\time_plot_means', dpi=100, bbox_inches= Bbox([[0,0],[cut_line,15]]))
        plt.savefig('D:\\final_outputs\\time_plot_vars', dpi=100, bbox_inches= Bbox([[cut_line,0],[12,15]]))
    else:
        plt.savefig('D:\\final_outputs\\time_plot', dpi=100)
    plt.show()


def kde_plot(subplots=True, stocks = None):
    if stocks is None:
        stocks = ["Suspended", "Sedimented", "Exported", "Total"]
    df = pd.read_excel("D:\\final_outputs\\pair_plot_df.xlsx")
    lines = ['dotted', 'dashed', 'dashdot', 'solid']
    if subplots:
        fig, ax = plt.subplots(ncols=4, figsize=(20,6))
        for i in range(len(stocks)):
            sns.kdeplot(data=df, x=stocks[i], hue="Mix",ax=ax[i], palette='tab10')
    else:
        fig,ax = plt.subplots(figsize=(8,8))
        for i in range(len(stocks)):
            sns.kdeplot(data=df, x=stocks[i], hue="Mix", ax=ax, palette='tab10', linestyle=lines[i])
    plt.savefig("D:\\final_outputs\\kde_plot", dpi=150)
    plt.show()




def plot_mp_mixes(volume_plot=True, marker_per_mix=True):
    from model_discrete_time import DataPreparationModule
    if marker_per_mix:
        markers = ['P', 'x', '*', 'd', 's', '.', 10, 11]
    else:
        markers = np.array(['o', 's', 'd', 'o', '*'])

    #markers = np.repeat(markers, 3)
    print(markers)
    colors = dp.get_colors(8)
    cmap = plt.get_cmap('tab10')
    # colors = [dp.get_hex(i) for i in colors]
    # colors = np.repeat(np.array(colors), 3)
    dat_prep = DataPreparationModule()
    if not volume_plot:
        fig, ax = plt.subplots(ncols=4, nrows=5, figsize=(20,20))
    else:
        fig, ax = plt.subplots(ncols=5, figsize=(25,6), sharey=True)
    chars = ['CSF', 'density', 'a']
    for i in range(8):
        file = f"D:\\server_files\\server_inputs\\mp_categories_server_{i}.xlsx"
        df = pd.read_excel(file)
        type_names =[f"wwtps_{i}_fraction" for i in ['fiber', 'fragment', 'bead', 'film', 'foam']]
        dic = {}
        for k in type_names:
            dic[k] = 1

        outcomes = dat_prep.create_mp_categories(file, type_factors=dic)
        dens = outcomes['occurrence']

        if volume_plot:
            for j in range(5):
                sort = np.argsort(df['a'][3 * j: 3 * (j + 1)].values).tolist()
                if marker_per_mix:
                    mark = markers[i]
                else:
                    mark = markers[j]
                ax[j].plot(df['a'][3 * j: 3 * (j + 1)].values[sort], np.cumsum(dens[3 * j: 3 * (j + 1)][sort]),
                           marker=mark, color=cmap(i),
                           markersize=10, label=f'Mix {i+1}')

                if i == 7:
                    ax[j].set_xlabel("a", fontsize=16 )
                    ax[j].set_ylabel('cummulative occurrence', fontsize=16)
                    ax[j].set_xscale('log')
                    ax[j].set_title(str(df['type'][3*j]), fontsize=16)
                    ax[j].legend()

        else:
            for j in range(5):
                dens[3*j:3*(j+1)] = sorted(dens[3*j:3*(j+1)])
            for p in range(4):
                if p == 0:
                    for j in range(5):
                        ax[j, p].plot([*range(3*j, 3*(j+1))],dens[3*j: 3*(j+1)], marker=markers[j], color=colors[i], markersize=10)
                        ax[j, p].set_yscale('log')
                else:
                    for j in range(5):
                        ax[j, p].plot(df[chars[p - 1]][3*j: 3*(j+1)], df[chars[p - 2]][3*j: 3*(j+1)], marker=markers[j],
                                      color=colors[i], markersize=10)
                        ax[j, p].set_ylabel(chars[p - 2])
                        ax[j, p].set_xlabel(chars[p - 1])

    plt.tight_layout()
    plt.savefig('D:\\final_outputs\\MPs_plot_a', dpi=150)
    plt.show()

def plot_mp_mixes_3d():
    from model_discrete_time import DataPreparationModule
    #markers = np.array([f'${i}$' for i in range(8)]) #np.array(['$f$', 's', 'd', 'o', '*'])
    markers = ['P', 'x', '*','d', 's', '.', 10, 11]
    #markers = np.repeat(markers, 3)
    print(markers)
    #colors = dp.get_colors(8)
    cmap = plt.get_cmap("tab10")
    # colors = [dp.get_hex(i) for i in colors]
    # colors = np.repeat(np.array(colors), 3)
    dat_prep = DataPreparationModule()
    fig, ax = plt.subplots(ncols=5, subplot_kw={'projection': "3d"}, figsize=(30,8))

    chars = ['CSF', 'density', 'a']
    for i in range(8):
        df = pd.read_excel(f"D:\\server_files\\server_inputs\\mp_categories_server_{i}.xlsx")
        if i == 0:
            print(df['type'])
        # dens = dat_prep.mp_shape_dist(df['CSF']) * dat_prep.mp_density_dist(df['density']) * dat_prep.mp_size_dist(df['a'])
        # dens = np.array(dens)
        # for j in range(5):
        #     dens[3*j:3*(j+1)] = sorted(dens[3*j:3*(j+1)])
        # for p in range(4):
        #     if p == 0:
        #         for j in range(5):
        #             ax[j, p].plot([*range(3*j, 3*(j+1))],dens[3*j: 3*(j+1)], marker=markers[j], color=colors[i], markersize=10)
        #             ax[j, p].set_yscale('log')

        for j in range(5):
            ax[j].set_ylabel(chars[0])
            ax[j].set_xlabel(chars[1])
            ax[j].set_zlabel(chars[2])
            sort = np.argsort(df[chars[1]][3*j: 3*(j+1)]).tolist()
            marke, _, _ = ax[j].stem(df[chars[0]][3*j: 3*(j+1)].values[sort], df[chars[1]][3*j: 3*(j+1)].values[sort], df[chars[2]][3*j: 3*(j+1)].values[sort])
                # , markerfmt=markers[i],
                #        linefmt='lightgrey')
            marke.set_markersize(10)
            marke.set_markerfacecolor(cmap(i))

    plt.tight_layout()
    plt.savefig('D:\\final_outputs\\MPs_plot_3d', dpi=150)
    plt.show()

def partial_correlation_plot(save_data=False, per_mix=True, n=4, plt_type='per_mp',
                             file_name="D:\\final_outputs\\full_mp_correlations_test", add_mixes=False,
                             y_ticks=None):

    outs = ["Suspended", "Sedimented", "Exported", "Total"]
    df = pd.read_excel("D:\\server_files\\server_inputs\\sample_server_24.xlsx")

    uncertainties = pd.concat([df] * n)
    uncert_names = list(uncertainties.columns[2:])

    if save_data:
        # print(uncertainties)
        num_uncert = len(uncertainties.columns) - 2
        uncert_names = list(uncertainties.columns[2:])

        out_df = pd.read_excel("D:\\final_outputs\\pair_plot_df.xlsx")
        print(out_df.columns)

        for i in uncert_names:
            out_df[i] = uncertainties[i].values
        out_df = (out_df - out_df.min()) / (out_df.max() - out_df.min())
        out_df.to_excel("D:\\final_outputs\\corrrelation_df.xlsx")
    else:
        out_df = pd.read_excel("D:\\final_outputs\\corrrelation_df.xlsx")
    # print(out_df)
    # uncert_names.append('Mix')
    if add_mixes:
        for i in range(n):
            uncert_names.append(f'Mix {i}')
            temp=np.zeros(n*24)
            temp[24*n:24*(n+1)] = 1
            out_df[f'Mix {i}'] = temp
    coefficients = np.zeros((len(outs) * n, len(uncert_names)))
    pvals = np.zeros_like(coefficients)
    if plt_type == 'per_mp':
        for r in range(n):
            for i in range(len(outs)):
                for j in range(len(uncert_names)):
                    temp = uncert_names.copy()
                    temp.remove(uncert_names[j])
                    co = pg.partial_corr(data=out_df.iloc[24 * r:24 * (r + 1)], y=outs[i], x=uncert_names[j],
                                         y_covar=temp, method='pearson')
                    # print(coefficients)
                    # print(co.shape)
                    coefficients[r * 4 + i, j] = co.values[0][1]
                    pvals[r * 4 + i, j] = co.values[0][3]
                    # print(co.values[0][1])
                    # print(co.values[0][3])
    elif plt_type == 'per_area':
        full_data = np.load("D:\\final_outputs\\total_agg_1.npy")
        mp_names = pd.read_excel("D:\\server_files\\server_inputs\\mp_categories_server_7.xlsx")['names'].tolist()
        output_names = [f"{i} {j}" for i in mp_names for j in outs]
        column_names = output_names.copy()
        column_names.extend(uncert_names)
        num_runs = 24*n

        dat = np.zeros((num_runs, len(column_names)))
        dat[:, -len(uncert_names):] = np.tile(uncertainties[uncert_names].iloc[:24].values, (n, 1))
        for i in range(4):
            for j in range(len(mp_names)):
                dat[:, i + 4 * j] = np.sum(full_data[:, -1, i, j, :], axis=-1)
        full_mp_df = pd.DataFrame(columns=column_names,
                                  index=[*range(num_runs)],
                                  data=dat
                                  )
        print(full_mp_df)
        full_mp_df = (full_mp_df - full_mp_df.min()) / (full_mp_df.max() - full_mp_df.min())
        coefficients = np.zeros((len(outs) * n * 5, len(uncert_names) * 3))
        pvals = np.zeros_like(coefficients)
        for mix in range(n):
            for mp in range(len(mp_names)):
                for u in range(len(uncert_names)):
                    for o in range(len(outs)):
                        temp = uncert_names.copy()
                        temp.remove(uncert_names[u])
                        co = pg.partial_corr(data=full_mp_df.iloc[24 * mix:24 * (mix + 1)],
                                             y=f"{mp_names[mp]} {outs[o]}",
                                             x=uncert_names[u],
                                             y_covar=temp)
                        coefficients[mix * len(outs) + o + len(outs) * n * int(np.floor(mp / 3)),
                                     len(uncert_names) * (mp % 3) + u] = co.values[0][1]
                        pvals[mix * len(outs) + o + len(outs) * n * int(np.floor(mp / 3)),
                              len(uncert_names) * (mp % 3) + u] = co.values[0][3]
    elif plt_type == 'full_mp':
        full_data = np.load("D:\\final_outputs\\total_agg_1.npy")
        mp_names = pd.read_excel("D:\\server_files\\server_inputs\\mp_categories_server_7.xlsx")['names'].tolist()
        output_names = [f"{i} {j}" for i in mp_names for j in outs]
        column_names = output_names.copy()
        column_names.extend(uncert_names)
        num_runs = 96

        dat=np.zeros((num_runs, len(column_names)))
        dat[:,-len(uncert_names):] = np.tile(uncertainties[uncert_names].iloc[:24].values, (n,1))
        for i in range(4):
            for j in range(len(mp_names)):
                dat[:,i+4*j] = np.sum(full_data[:,-1,i,j,:], axis=-1)
        full_mp_df = pd.DataFrame(columns=column_names,
                                  index=[*range(num_runs)],
                                  data=dat
                                  )
        print(full_mp_df)
        full_mp_df = (full_mp_df - full_mp_df.min()) / (full_mp_df.max() - full_mp_df.min())
        coefficients = np.zeros((len(outs)*n*5,len(uncert_names)*3))
        pvals = np.zeros_like(coefficients)
        for mix in range(n):
            for mp in range(len(mp_names)):
                for u in range(len(uncert_names)):
                    for o in range(len(outs)):
                        temp = uncert_names.copy()
                        temp.remove(uncert_names[u])
                        co = pg.partial_corr(data=full_mp_df.iloc[24 * mix:24 * (mix + 1)],
                                             y=f"{mp_names[mp]} {outs[o]}",
                                             x=uncert_names[u],
                                             y_covar=temp)
                        coefficients[mix * len(outs) + o + len(outs)*n*int(np.floor(mp/3)),
                                     len(uncert_names) * (mp % 3) + u] = co.values[0][1]
                        pvals[mix * len(outs) + o + len(outs)*n*int(np.floor(mp/3)),
                                     len(uncert_names) * (mp % 3) + u] = co.values[0][3]

        #
    elif plt_type == 'small':
        coefficients = np.zeros((len(outs), len(uncert_names)))
        pvals = np.zeros_like(coefficients)
        for i in range(len(outs)):
            for j in range(len(uncert_names)):
                temp = uncert_names.copy()
                temp.remove(uncert_names[j])
                if i != 3:
                    co = pg.partial_corr(data=out_df, y=outs[i], x=uncert_names[j],
                                         y_covar=temp, method='pearson')
                else:
                    co = pg.partial_corr(data=out_df[0:24], y=outs[i], x=uncert_names[j],
                                         y_covar=temp, method='pearson')
                # print(coefficients)
                # print(co.shape)
                coefficients[i, j] = co.values[0][1]
                pvals[i, j] = co.values[0][3]
    else:
        return
    print(pvals)
    key_1 = np.where(pvals > 0.05)
    #key_almost = np.where((pvals <= 0.1) & (pvals > 0.05))
    key_2 = np.where((pvals <= 0.05) & (pvals > 0.01))
    key_3 = np.where((pvals <= 0.01) & (pvals > 0.001))
    key_4 = np.where(pvals <= 0.001)
    texts = np.empty_like(pvals, dtype='<U5')
    texts[key_1] = ''
    #texts[key_almost] = '-'
    texts[key_2] = '*'
    texts[key_3] = '**'
    texts[key_4] = '***'

    if plt_type == 'full_mp':
        figsize = (25,15)
        y_names = [j for _ in range(3) for j in uncert_names]
        x_names = [outs[i] for _ in range(5) for j in range(n) for i in range(len(outs))]
        shrink=0.1
    elif plt_type == 'small':
        figsize = (5, 8)
        y_names = [j for j in uncert_names]
        x_names = [i for i in outs]
        shrink = 0.8
    else:
        figsize=(18,8)
        shrink=0.5
        y_names = uncert_names
        x_names = [outs[i] for _ in range(n) for i in range(len(outs))]
    fig, ax = plt.subplots(figsize=figsize)

    df = pd.DataFrame(coefficients.T, y_names, x_names)
    cmap = cm.get_cmap('PiYG')
    im = ax.imshow(df, cmap=cmap, vmin=-1, vmax=1)  # annot=pvals.T)## linewidths=1, linecolor='w')
    if plt_type == 'per_mp' or plt_type == 'full_mp':
        ax.vlines([-0.5 + 4 * i for i in range(1,int( len(x_names) /4))], ymin=-0.5, ymax=len(y_names) - 0.5,
                  colors='black')
        if plt_type == 'per_mp':
            for i in range(n):
                plt.figtext(0.15 + i*0.0875, 0.918, f"Mix {i+1}")
        if plt_type == 'full_mp':
            ax.vlines([-0.5 + len(outs) * n * i for i in range(1,5)], ymin=-0.5, ymax=len(y_names) - 0.5, colors='w',
                      linewidth=2)
    if plt_type == 'small':
        ax.vlines(2.5, ymin=-0.5, ymax=len(y_names) -0.5, linestyles='dashed', colors='black')
    for i in range(len(x_names)):
        for j in range(len(y_names)):
            ax.annotate(texts[i, j], xy=(i, j), ha='center', va='center')
            if texts[i,j] != '':
                ax.annotate(np.round(coefficients[i,j],2), xy=(i,j+0.35), ha='center', va='center', fontsize=8)
    # plt.imshow(coefficients.T, cmap='Spectral_r', vmin=-1, vmax=1)
    if y_ticks is None:
        y_ticks = y_names
    ax.set_yticks(np.arange(len(y_names)), y_ticks)
    ax.set_xticks(np.arange(len(x_names)), x_names, rotation=90)
    fig.colorbar(im, shrink=shrink)
    plt.tight_layout()
    plt.savefig(file_name, dpi=100)
    plt.show()

if __name__ == '__main__':
    #This is the start of the uncertainty analysis plot.
    #Similar to what we did during agent based modelling.
    # total_data = np.load("D:\\final_outputs\\total_agg_1.npy")
    # pair_df = pd.DataFrame({"Suspended": np.sum(total_data[:, -1, 0, :, :], axis=(-1, -2)),
    #                         "Sedimented": np.sum(total_data[:, -1, 1, :, :], axis=(-1, -2)),
    #                         "Sinks": np.sum(total_data[:, -1, 2, :, :], axis=(-1, -2)),
    #                         "Total": np.sum(total_data[:, -1, 3, :, :], axis=(-1, -2)),
    #                         "Mix": np.array([int(i / 24) for i in range(96)])
    #                         })
    num_mixes = 8
    #
    # plot_mp_mixes()
    #save_sample_table()

    # ALL PLOTS

    uncertainties = pd.read_excel("D:\\server_files\\server_inputs\\sample_server_24.xlsx")
    # uncert_names = list(uncertainties.columns[2:])
    # print(uncert_names)
    uncert_names = ['wwtps_rem_eff_secondary','wwtps_rem_eff_advanced','wwtps_washing_mp', 'a7', 'beta1', 'beta2']
    data = None
    uncertainty_scatter_plot(data,uncertainties,uncert_names,concat_data=True, percentage=True, plot_log=False,
                             out_file= "D:\\final_outputs\\uncertainty_scatter_small_percent.png", n_mix=num_mixes,
                             small_plot=True, legend=True)
    # data = np.load("D:\\final_outputs\\total_agg_1.npy")
    # uncertainty_scatter_plot(data, uncertainties, uncert_names, concat_data=False, percentage=True, plot_log=False,
    #                          out_file="D:\\final_outputs\\uncertainty_scatter_percent.png", n_mix=num_mixes)
    # create_pair_plot(save_data=True, n_mix=num_mixes)
    # kde_plot(subplots=True, stocks=['Suspended', 'Sedimented', "Exported"])
    # uncert_labels = [r'$\gamma_7$', r'$\gamma_8$', r'$\beta_1$',r'$\beta_2$',r'$\beta_3$',r'$\beta_4$',
    #                  r'WWTPs $r_{eff}$ primary', r'WWTPs $r_{eff}$ secondary', r'WWTPs $r_{eff}$ advanced',
    #                   r'WWTPs $N_{MP, wash}$','WWTPs fiber fraction', 'WWTPs fragment fraction','WWTPs film fraction',
    #                  'WWTPs bead fraction', 'WWTPs foam fraction']
    # partial_correlation_plot(save_data=True,plt_type='per_mp', add_mixes=False,
    #                          file_name="D:\\final_outputs\\mp_correlations_full", n=num_mixes,
    #                          y_ticks=uncert_labels)
    #total_time_plot(per_mix=True, plt_type='small', n_mix=num_mixes)

