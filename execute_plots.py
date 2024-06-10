import matplotlib.pyplot as plt
import xarray as xr
from geo_analysis import create_bulk_basin_plots, plot_chart_map


def call_bulk_basin_plots():
    areas = [
        'af',  # Africa
        'ar',  # North American Arctic
        'as',  # Central and South - East Asia
        'au',  # Australia and Oceania
        'eu',  # Europe and Middle East
        'gr',  # Greenland
        'na',  # North America and Caribbean
        'sa',  # South America
        'si'  # Siberia
    ]
    # shape_files = [f'D:\\hybas_{areas[0]}_lev01-12_v1c\\hybas_{areas[0]}_lev02_v1c.shp']
    # 'D:\\hybas_af_lev01-12_v1c\\hybas_af_lev02_v1c.shp']
    world_basins = [f'D:\\world_basins\\hybas_{areas[i]}_lev01-12_v1c\\hybas_{areas[i]}_lev02_v1c.shp'
                    for i in range(len(areas))]
    data_file = 'D:\\test_yu_8cat_1000.nc'
    geo_data = xr.open_dataset(data_file)
    flow_map = xr.open_dataset('D:\\inputs\\channel_parameters_extended.nc')['lddMap']
    flow_map = flow_map.values
    mp_names = [
        "fiber",
        "fiber_large",
        "plate_thin",
        "plate",
        "disk_thin",
        "disk_thick",
        "cylinder",
        "prism",
    ]
    create_bulk_basin_plots(mp_names, world_basins, flow_map, geo_data)

areas = [
        'af',  # Africa
        'ar',  # North American Arctic
        'as',  # Central and South - East Asia
        'au',  # Australia and Oceania
        'eu',  # Europe and Middle East
        'gr',  # Greenland
        'na',  # North America and Caribbean
        'sa',  # South America
        'si'  # Siberia
    ]
# shape_files = [f'D:\\hybas_{areas[0]}_lev01-12_v1c\\hybas_{areas[0]}_lev02_v1c.shp']
# 'D:\\hybas_af_lev01-12_v1c\\hybas_af_lev02_v1c.shp']
world_basins = [f'D:\\world_basins\\hybas_{areas[i]}_lev01-12_v1c\\hybas_{areas[i]}_lev02_v1c.shp'
                for i in range(len(areas))]
data_file = 'D:\\test_yu_8cat_1000.nc'
geo_data = xr.open_dataset(data_file)
flow_map = xr.open_dataset('D:\\inputs\\channel_parameters_extended.nc')['lddMap']
flow_map = flow_map.values
mp_names = [
    "fiber",
    "fiber_large",
    "plate_thin",
    "plate",
    "disk_thin",
    "disk_thick",
    "cylinder",
    "prism",
]
fig, ax = plt.subplots(figsize=(30,15))
plot_chart_map(world_basins, geo_data, mp_names, [], flow_map, ax,
                   time_step=-1,
                   prefixes=['sed_mp_', 'sus_mp_', 'ocean_mp_'],
                   explode=True, colors=['orange', "cornflowerblue", "mediumorchid"],
                   bound=False,
                   charts_outline=True,
                   poly_edge_width=0.5,
                   chart_type='pie_stocks',
                   colmap="Greens",
                   radius=3,
                   plot_borders=True,
                   type_names=["Sediment", "River", "Sinks"]
                   )
ax.set_xlim(-181, 181)
ax.set_ylim(-70, 91)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig("D:\\out_tests\\global_stocks_pies.png", dpi=400)
plt.show()