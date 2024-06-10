import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
import rasterstats as rstats
import xarray as xr
from rasterio.transform import from_origin
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogFormatter
import netCDF4 as nc


# print(directions.shape)

def filter_ocean_and_river_stocks(raster, raster_keys, flow_map, t_step, print_total=True):
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
    if print_total:
        sed = np.sum(out_dict["sed_mp_total"])
        sus = np.sum(out_dict["sus_mp_total"])
        sink = np.sum(out_dict["ocean_mp_total"])
        print(f'total mps in the world: {sed + sus + sink}')
        print(f"sink fraction: {sink/ (sed+sus+sink)}")
        print(f"sed fraction: {sed/ (sed+sus+sink)}")
        print(f"sus fraction: {sus/ (sed+sus+sink)}")
    return out_dict


def get_data_pie_chart_plot(areas, rasters, raster_keys, explode=True,
                            affine=from_origin(-180, 90, 0.083333333333333, 0.083333333333333), cutoff=1,
                            crs=4326,
                            land_type='poly'):
    if land_type == 'poly':
        centroids = areas.to_crs(crs).representative_point()
        centroids.index = [*range(len(centroids))]
        data_dict = {'labels': raster_keys,
                     'total_sizes': np.zeros(len(areas))}
        start = True
        polygon_keys = {}
        for j in range(len(raster_keys)):

            data = rasters[raster_keys[j]]

            polygon_index = 0
            if j == 1:
                start = False
            for i in range(len(areas)):
                if areas.iloc[i].geometry.area > cutoff:
                    n = f'polygon{i}'
                    if start:
                        x_coord = centroids.x[i]
                        y_coord = centroids.y[i]
                        data_dict[n] = {'coords': [x_coord, y_coord],
                                        'stats': {}}
                        polygon_keys[polygon_index] = n

                    if 'ocean' in raster_keys[j]:
                        all_touch = True
                    else:
                        all_touch = False
                    stat = rstats.zonal_stats(areas.iloc[i].geometry, data, affine=affine,
                                              stats='sum', all_touched=all_touch)
                    if stat[0]['sum'] is None:
                        stat[0]['sum'] = 0
                    data_dict[n]['stats'][raster_keys[j]] = stat[0]['sum']
                    data_dict['total_sizes'][polygon_index] += stat[0]['sum']
                    polygon_index += 1
    elif land_type == 'raster':
        num_basins = 53
        lats = areas["lat"][:]
        lons = areas['lon'][:]
        world_mask = areas["mask5minFromTop_map"][:, :].astype(int)
        centers = np.zeros((num_basins, 2))
        data_dict = {'labels': raster_keys,
                     'total_sizes': np.zeros(num_basins)}
        polygon_keys = {}
        for i in range(num_basins):
            n = f'polygon{i}'
            centers[i, 0] = np.average(lons[np.where(world_mask == i + 1)[1]])
            centers[i, 1] = np.average(lats[np.where(world_mask == i + 1)[0]])
            # manually adjust those that don't correspond exactly
            if i == 17: #Laos + indonesia
                centers[i, 0] -= 4
                centers[i, 1] -= 9
            if i == 18:
                centers[i, 0] -= 3
                centers[i, 1] += 1
            if i == 21:
                centers[i, 0] -= 1
                centers[i, 1] -= 4
            if i == 44:
                centers[i, 1] += 2
            if i == 45:
                centers[i, 0] += 3
            if i == 47:
                centers[i, 1] -= 4

            data_dict[n] = {'coords': [centers[i, 0], centers[i, 1]],
                            'stats': {}}
            polygon_keys[i] = n
            for j in range(len(raster_keys)):
                data = rasters[raster_keys[j]]
                stat = np.nansum(data[np.where(world_mask == i + 1)])
                data_dict[n]['stats'][raster_keys[j]] = stat
                data_dict['total_sizes'][i] += stat

                # if j == len(raster_keys) -1:
                #     data_dict['total_sizes'].append(sum(data_dict[n]['stats']))
    # Data dict contains the summed values
    else:
        return

    return data_dict, polygon_keys


def plot_chart_map(shape_files, raster_data, names, used_keys, flow_dir, ax,
                   time_step=-1,
                   prefixes=['sus_mp_', 'sed_mp_', 'ocean_mp_'],
                   explode=True, colors=None,
                   bound=False,
                   charts_outline=True,
                   poly_edge_width=0.5,
                   chart_type='pie',
                   colmap="Blues",
                   radius=3,
                   plot_borders=True,
                   type_names=None,
                   land_type='raster',
                   outlines_only=False,
                   plot_variances=None,
                   plot_total=None
                   ):
    """Function that plots a map of all the passed shape objects, and calculates totals of different stocks and
    represents those as a (pie) chart."""
    print(colmap)
    if land_type == 'poly':
        if type(shape_files) is not list:
            land = gpd.read_file(shape_files)
        else:
            gdfs = []
            for i in shape_files:
                gdfs.append(gpd.read_file(i))
            if len(gdfs) > 1:
                land = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            else:
                land = gdfs[0]
        if explode:
            land = land.explode(index_parts=True)
        land.index = [*range(len(land))]

    elif land_type == 'raster':
        land = nc.Dataset(shape_files)
    else:
        return
    if not outlines_only:
        keys = [prefixes[j] + names[i] for i in range(len(names)) for j in range(len(prefixes))]
        print(keys)
        data = filter_ocean_and_river_stocks(raster_data, keys, flow_dir, time_step)
        # print(land)

        pie_data, poly_keys = get_data_pie_chart_plot(land, data, keys, land_type=land_type)
        poly_index = [int(poly_keys[i][7:]) for i in range(len(poly_keys))]
        if colors is None:
            colors = [plt.cm.Paired(i) for i in range(len(used_keys))]

        if chart_type == 'pie':
            total_lengths = []
            print(f'number of polygons: {len(poly_keys)}')
            for i in range(len(poly_keys)):
                temp = []
                for j in used_keys:
                    # print(pie_data[poly_keys[i]]['stats'])
                    temp.append(pie_data[poly_keys[i]]['stats'][j])
                pie_data[poly_keys[i]]['used_stats'] = temp
                total_lengths.append(sum(temp))
            print(len(total_lengths))
            total_lengths = np.array(total_lengths)
        elif chart_type == "pie_stocks":
            total_lengths = []
            for i in range(len(poly_keys)):
                total_lengths.append(pie_data["total_sizes"][i])
                temp = [0 for j in range(len(prefixes))]
                for j in range(len(prefixes)):
                    temp[j] = sum([pie_data[poly_keys[i]]['stats'][f"{prefixes[j]}{names[k]}"]
                                   for k in range(len(names))])
                pie_data[poly_keys[i]]['used_stats'] = np.array(temp)
            total_lengths = np.array(total_lengths)
        elif chart_type == 'variance':
            total_lengths = []
        else:
            total_lengths = []

    if plot_borders:
        edge_col = 'darkgrey'
    else:
        edge_col = 'black'
    if land_type == 'poly':
        land['total_mp'] = 0
        land.iloc[poly_index, land.columns.get_loc('total_mp')] = total_lengths
        land.to_crs(4326).plot(color="lightgrey",  # loc[~land.index.isin(poly_index)]
                               edgecolor='black', ax=ax,
                               linewidth=poly_edge_width,
                               zorder=-5)
        ax_temp = land.iloc[poly_index].to_crs(4326).plot(column='total_mp', cmap=colmap, legend=True,
                                                          legend_kwds={"orientation": "horizontal", "pad": 0.04,
                                                                       'fraction': 0.046},
                                                          norm=LogNorm(
                                                              vmin=min(total_lengths[np.nonzero(total_lengths)]),
                                                              vmax=max(total_lengths)),
                                                          edgecolor='black', ax=ax, linewidth=poly_edge_width,
                                                          zorder=-4)
        fig_ = ax_temp.figure
        cb = fig_.axes[1]
        #cb.tick_params(labelsize=16)

    elif land_type == 'raster':
        world_mask = land["mask5minFromTop_map"][:, :].astype(int)
        #print('unique sinks')
        #print(np.unique(world_mask))
        lats = land['lat'][:]
        lons = land['lon'][:]
        data = np.zeros_like(world_mask).astype(np.float32)
        for i in range(53):
            where = np.where(world_mask == i + 1)
            if not outlines_only:
                data[where[0], where[1]] = total_lengths[i]
            #print(f"combined result {i} is equal to {np.sum(data)} (element {total_lengths[i]})")
        #print('unique datapoints:')
        #print(np.unique(data))
        mdata = np.ma.masked_where(data <= 0, data)

        X, Y = np.meshgrid(lons, lats)
        #print(min(total_lengths[np.nonzero(total_lengths)]))

        unique_data = np.unique(data)
        #print(unique_data)
        rest_data = np.where((data == 0) & (world_mask > 0), 0, 1)
        grayscale = 0.8274509803921568
        rest_data = np.ma.masked_array(np.ones_like(rest_data)*grayscale, rest_data)
        ax.contour(X, Y, world_mask, colors='black', linewidths=(poly_edge_width,),
                   zorder=-3, levels=np.unique(world_mask[np.where(world_mask > 0)]), extend='both'
                   )
        ax.pcolormesh(X, Y, rest_data, cmap='gray', vmin=0,vmax=1,
                   zorder=-3.5
                   )
        #ax.contourf(X, Y, background, levels=[0, 1], colors=('white', 'lightgrey'), zorder=-5)
        if not outlines_only:
            cs = ax.contourf(X, Y, mdata, norm=LogNorm(vmin=min(total_lengths[np.nonzero(total_lengths)]),
                                                       vmax=max(total_lengths)),
                             cmap=colmap,
                             zorder=-4, levels=unique_data[np.where(unique_data > 0 )]
                             )
            # lowest_power = np.ceil(np.log10(np.min(mdata)))
            # highest_power = np.floor(np.log10(np.nanmax(mdata)))
            formatter = LogFormatter(10, labelOnlyBase=False, minor_thresholds=(0,0))

            cb = ax.figure.colorbar(cs,ticks=[10**x for x in range(12,16)],
                                    orientation="horizontal", pad=0.02, fraction=0.046, format=formatter)
            #cb.set_ticks()
            cb.ax.tick_params(labelsize=16)

    if plot_borders:
        world = gpd.read_file("D:\\world_basins\\50m_cultural\\ne_50m_admin_0_countries.shp")
        # print(world.columns)
        # print(world)
        world = world[world["SOVEREIGNT"] != "Antarctica"]
        # print(world[world["SOVEREIGNT"] == "Greenland"])
        world.to_crs(4326).boundary.plot(edgecolor='black', linewidth=poly_edge_width * 0.5, ax=ax,
                                         zorder=-2)
        if land_type == 'raster':
            world[world["NAME"] == "Greenland"].to_crs(4326).plot(color='lightgrey', ax=ax, zorder=-3,
                                                                  edgecolor='black',
                                                                  linewidth=poly_edge_width)

    if not outlines_only:
        for i in range(len(poly_keys)):
            # print(pie_data[poly_keys[i]]['used_stats'])
            if sum(pie_data[poly_keys[i]]['used_stats']) != 0:
                if 'pie' in chart_type:

                    offset = 4.5
                    cent = (pie_data[poly_keys[i]]['coords'][0], pie_data[poly_keys[i]]['coords'][1] + offset)
                    rad = radius  # max(5 * total_lengths[i] / max(total_lengths), 1)
                    ax.pie(pie_data[poly_keys[i]]['used_stats'],
                           center=cent,
                           radius=rad,
                           colors=colors,
                           frame=True,
                           )
                    if charts_outline:
                        ax.add_patch(plt.Circle(cent, rad + 0.08, fill=False, edgecolor='w', linewidth=1))
                    ax.add_patch(plt.Polygon([[cent[0], cent[1] - offset], [cent[0] + 0.5, cent[1] - offset + 0.75],
                                              [cent[0] - 0.5, cent[1] - offset + 0.75]], color='w'))

                    if plot_variances is not None:
                        bar_width = 1
                        scale_bar = rad*1.4
                        bar_x =[cent[0] - rad - 0.5 -bar_width - bar_width*1.3 * j for j in range(len(plot_variances[:,0]))]
                        ax.bar(bar_x,
                               plot_variances[:,i]*scale_bar, bottom= cent[1] - rad, width=bar_width, linewidth=0.8, align='edge',
                               color=colors, edgecolor='w')
                        for j in range(len(plot_variances[:, i])):
                            if plot_variances[j, i] != np.nan:
                                heights = np.arange(stop=plot_variances[j, i],step=0.5)
                                if len(heights) > 1:
                                    plt.hlines(heights[1:]*scale_bar + cent[1] - rad, bar_x[j], bar_x[j] + bar_width,
                                               colors='w', linewidth=0.8)
        if plot_total is not None:
            plt.annotate('Global:', (65,-27),fontsize=18)
            xy = np.array([[100,-40],[100,-20],[63,-20],[63,-40]])
            ax.add_patch(plt.Polygon(xy, facecolor='lightgrey', edgecolor='black', linewidth=3))
            cent = (90, -30)
            rad = 8  # max(5 * total_lengths[i] / max(total_lengths), 1)
            ax.pie(plot_total[0,:3],
                   center=cent,
                   radius=rad,
                   colors=colors,
                   frame=True,
                   )
            if charts_outline:
                ax.add_patch(plt.Circle(cent, rad + 0.08, fill=False, edgecolor='w', linewidth=2))
            if plot_variances is not None:
                bar_width = 2.5
                scale_bar = rad * 1.2
                bar_x = [cent[0] - rad - 0.8 - bar_width - bar_width * 1.3 * j for j in range(len(plot_total[1, :]))]
                ax.bar(bar_x,
                       plot_total[1, :] * scale_bar, bottom=cent[1] - rad, width=bar_width, linewidth=1.6, align='edge',
                       color=colors, edgecolor='w')
                for j in range(len(plot_total[1, :])):
                    if plot_variances[1, j] != np.nan:
                        heights = np.arange(stop=plot_total[1, j], step=0.5)
                        if len(heights) > 1:
                            plt.hlines(heights[1:] * scale_bar + cent[1] - rad, bar_x[j], bar_x[j] + bar_width,
                                       colors='w', linewidth=1.6)

    if type_names is None:
        type_names = used_keys

    if not outlines_only:
        custom_lines = [Patch(facecolor=colors[i],
                              label=type_names[i]) for i in range(len(type_names))]
        ax.legend(handles=custom_lines, loc='lower left', fontsize=18)
    if bound:
        minx, miny, maxx, maxy = land.geometry.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    return


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def create_bulk_basin_plots(names,
                            shape_files,
                            directions,
                            nc_ds,
                            pref=['sed_mp_', 'sus_mp_', 'ocean_mp_'],
                            mapcols=['Oranges', 'Blues', 'YlGn'],
                            mins=[0.3, 0.2, 0.1],
                            ):
    mapcols = [truncate_colormap(plt.get_cmap(mapcols[i]), minval=mins[i], n=200) for i in range(len(mapcols))]
    # linewidths = [0.1,0.5,1]
    for k in range(len(pref)):
        keys_to_plot = [pref[k] + names[i] for i in range(len(names))]
        fig, axis = plt.subplots(figsize=(30, 15))
        # ax.set_box_aspect(1)
        plot_chart_map(shape_files, nc_ds, names, keys_to_plot, directions, axis, poly_edge_width=0.5,
                       colmap=mapcols[k], type_names=names)
        axis.set_xlim(-181, 181)
        axis.set_ylim(-70, 91)
        axis.set_xticks([])
        axis.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'D:\\out_tests\\pie_chart_map_{pref[k]}_yu_1000.png', dpi=400)
        plt.show()


def plot_river_map(plt_type,
                   data_file='D:\\test_yu_8cat_1000.nc',
                   flow_file='D:\\inputs\\channel_parameters_extended.nc',
                   plot_var="sed_mp_clinder", plot_t=-1,
                   world_file="D:\\world_basins\\50m_cultural\\ne_50m_admin_0_countries_lakes.shp",
                   out_file="D:\\out_tests\\test_river_plot.png",
                   xslice=slice(None),
                   yslice=slice(None),
                   colmap='YlOrBr'
                   ):
    nc_ds = xr.open_dataset(data_file)
    times = nc_ds['time'].values
    lats = nc_ds['latitude'].values
    lons = nc_ds['longitude'].values

    flow_map = xr.open_dataset(flow_file)['lddMap']
    flow_map = flow_map.values[xslice, yslice]
    fig, ax = plt.subplots(figsize=(30, 15))
    world = gpd.read_file(world_file)
    # print(world.columns)
    # print(world)
    world = world[world["SOVEREIGNT"] != "Antarctica"]
    world.to_crs(4326).plot(color='lightgray', edgecolor='black', linewidth=0.5, ax=ax,
                            zorder=-2)
    if 'rivers' in plt_type:
        if type(plot_var) == str:
            nc_arr_vals = nc_ds[plot_var].values[plot_t, xslice, yslice]
        else:
            nc_arr_vals = np.zeros_like(flow_map)
            for var in plot_var:
                nc_arr_vals += nc_ds[var].values[plot_t, xslice, yslice]
        Z = nc_arr_vals
        if 'sink_bars' in plt_type:
            max_val = np.nanmax(Z)
            Z[np.where(flow_map == 5)] = 0
        else:
            Z[np.where(flow_map == 5)] = 0
            max_val = np.nanmax(Z)
        print(f'min: {np.nanmin(Z)}, max: {np.nanmax(Z)}')
        print(Z.shape)

        plt.pcolormesh(lons[xslice], lats[yslice], Z, norm=LogNorm(vmin=10,
                                                                   vmax=max_val), cmap=colmap)
        cbar = plt.colorbar(orientation="horizontal", pad=0.04, fraction=0.046)
        cbar.ax.tick_params(labelsize=16)

    if "sink_bars" in plt_type:
        if type(plot_var) == list:
            cutoff = 1e5
            nc_arr_vals = np.zeros_like(flow_map)
            for var in plot_var:
                nc_arr_vals += nc_ds[var].values[plot_t, xslice, yslice]
            Z = nc_arr_vals
            max_val = np.nanmax(Z)
            Z[np.where(flow_map != 5)] = 0
            Z[np.where(Z < cutoff)] = 0
            inds = np.nonzero(Z)
            print(np.nanmax(Z))

            norm = LogNorm(vmin=10, vmax=max_val)

            # Create a ScalarMappable object
            sm = plt.cm.ScalarMappable(cmap=colmap, norm=norm)
            cols = sm.to_rgba(Z[inds])
            # sm.set_array(Z[inds])
            print(cols)
            vals = np.log(Z[inds])
            vals = (vals - np.log(cutoff)) / (np.nanmax(vals) - np.log(cutoff))
            ax.bar(lons[inds[1]], 1.6 * vals, bottom=lats[inds[0]],
                   # norm=LogNorm(vmin=10, vmax=max_val),
                   color=cols, width=0.19, edgecolor='black', linewidth=0.1
                   )
        else:
            raise TypeError
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-181, 181)
    ax.set_ylim(-70, 91)

    plt.tight_layout()
    plt.savefig(out_file, dpi=450)
    plt.show()

def load_spatial_totals(t_step = -1, n_mix=8, n_runs=24):
    total_dat = np.zeros((n_mix*n_runs, 4, 53))
    ind = 0
    for m in range(n_mix):
        for i in range(24):
            data = np.load(f"D:\\final_outputs\\agg_res\\mix_{m}\\agg_res_{m}_{i}.npy")
            total_dat[ind,:3,:] = np.sum(data[t_step, :, :, :], axis=1)
            ind += 1
    total_dat[:,3,:] = np.sum(total_dat[:,:3,:], axis=1)
    return total_dat

def variance_subplot(variances = None, areas="D:\\world_basins\\pcr_basins\\mask5minFromTop.nc", plt_countries=True):
    # totals = np.load("D:\\final_outputs\\totals\\mean_total.npy")
    # print(totals.shape)
    # dat = {
    #     'sus_mp_total': np.zeros((1, 2160, 4320)),
    #     'sed_mp_total': np.zeros((1, 2160, 4320))
    # }
    # dat['sus_mp_total'][0] = totals[0, :, :]
    # dat['sed_mp_total'][0] = totals[1, :, :]
    # keys_to_plot = ['sus_mp_' + names[i] for i in range(len(names))]
    areas = nc.Dataset(areas)
    titles=['Suspended', 'Sedimented', 'Exported', 'Total']
    if variances == None:
        spat_dat = load_spatial_totals()
        variances = np.nanstd(spat_dat, axis=0) / np.nanmean(spat_dat, axis=0)
        variances = np.nan_to_num(variances, copy=False)
        print(spat_dat.shape)
    # spat_tot = np.nansum(spat_dat, axis=-1)
    # global_values = np.zeros((2, 4))
    # global_values[0, :] = np.nanmean(spat_tot[:, :], axis=0)
    # global_values[1, :] = np.nanstd(spat_tot[:, :], axis=0) / np.nanmean(spat_tot[:, :], axis=0)

    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(30,15), layout='compressed')
    for p in range(4):
        inds = [(0,0), (0,1),(1,0),   (1,1)]

        world_mask = areas["mask5minFromTop_map"][:, :].astype(int)
        # print('unique sinks')
        # print(np.unique(world_mask))
        lats = areas['lat'][:]
        lons = areas['lon'][:]
        data = np.zeros((len(lons),len(lats)))
        data = np.zeros_like(world_mask).astype(np.float32)
        for i in range(53):
            where = np.where(world_mask == i + 1)
            data[where[0], where[1]] = variances[p, i]
            # print(f"combined result {i} is equal to {np.sum(data)} (element {total_lengths[i]})")
        # print('unique datapoints:')
        # print(np.unique(data))
        mdata = np.ma.masked_where(data <= 0, data)

        X, Y = np.meshgrid(lons, lats)
        # print(min(total_lengths[np.nonzero(total_lengths)]))

        unique_data = np.unique(data)
        # print(unique_data)
        rest_data = np.where((data == 0) & (world_mask > 0), 0, 1)
        grayscale = 0.8274509803921568
        rest_data = np.ma.masked_array(np.ones_like(rest_data) * grayscale, rest_data)
        ax[inds[p]].contour(X, Y, world_mask, colors='black', linewidths=(1,),
                            zorder=-3, levels=np.unique(world_mask[np.where(world_mask > 0)]), extend='both'
                            )
        ax[inds[p]].pcolormesh(X, Y, rest_data, cmap='gray', vmin=0, vmax=1,
                               zorder=-3.5
                               )

        if plt_countries:
            world = gpd.read_file("D:\\world_basins\\50m_cultural\\ne_50m_admin_0_countries.shp")
            # print(world.columns)
            # print(world)
            world = world[world["SOVEREIGNT"] != "Antarctica"]
            # print(world[world["SOVEREIGNT"] == "Greenland"])
            world.to_crs(4326).boundary.plot(edgecolor='black', linewidth=0.5, ax=ax[inds[p]],
                                             zorder=-2, aspect='equal')
            world[world["NAME"] == "Greenland"].to_crs(4326).plot(color='lightgrey', ax=ax[inds[p]], zorder=-3,
                                                                  edgecolor='black',
                                                                  linewidth=0.5,aspect='equal')

        # ax.contourf(X, Y, background, levels=[0, 1], colors=('white', 'lightgrey'), zorder=-5)
        colmap = 'Reds'
        cs = ax[inds[p]].contourf(X, Y, mdata, vmin=0, vmax=np.max(variances),
                                  cmap=colmap,
                                  zorder=-4, #levels=unique_data[np.where(unique_data > 0)]
                                  )
        # lowest_power = np.ceil(np.log10(np.min(mdata)))
        # highest_power = np.floor(np.log10(np.nanmax(mdata)))
        ax[inds[p]].set_title(titles[p], fontsize=18)
        #ax[inds[p]].set_box_aspect(aspect=0.5)
        ax[inds[p]].set_xlim(-181, 181)
        ax[inds[p]].set_ylim(-70, 91)
        ax[inds[p]].set_xticks([])
        ax[inds[p]].set_yticks([])
        if p == 1:
            #formatter = LogFormatter(10, labelOnlyBase=False, minor_thresholds=(0, 0))

            cb = fig.colorbar(cs, ax=ax[:,:], location='bottom', shrink=0.6,
                              pad=0.05) #extend='max') #orientation="horizontal", pad=0.02, fraction=0.046,
            # cb.set_ticks()
            cb.ax.tick_params(labelsize=16)
    #plt.tight_layout()
    plt.savefig('D:\\final_outputs\\variance_maps', dpi=500)
    plt.show()


if __name__ == '__main__':
    # names = [
    #     'fiber_l',
    #     'fiber_s',
    #     'frag_s',
    #     'frag_l',
    #     'film_d',
    #     'film_p',
    #     'foam_l',
    #     'foam_s',
    #     'bead_l',
    #     'bead_s'
    #
    # ]
    # names = [
    #     "fiber",
    #     "fiber_large",
    #     "plate_thin",
    #     "plate",
    #     "disk_thin",
    #     "disk_thick",
    #     "cylinder",
    #     "prism",
    # ]

    # plot_river_map('sink_bars_rivers', plot_var=['sed_mp_cylinder','sus_mp_cylinder'],
    #                out_file="D:\\sink_bars_test.png")
    variance_subplot()
