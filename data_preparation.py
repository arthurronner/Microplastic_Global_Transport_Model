import netCDF4 as nc
import numpy as np
import time
import pickle
from model_discrete_time import MPGTModel
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from openpyxl import load_workbook
directory = "C:\\Users\\Arthur\\Documents\\uni bestanden\\Industrial Ecology\\0. master thesis\\model_input\\"
filename = "channel_parameters_extended.nc"
def construct_basins():


    inputs = nc.Dataset(directory + filename, "r")
    flow_map = np.ma.getdata(inputs["lddMap"][:,:]).copy()
    inputs.close()
    flow_map[np.where(flow_map == 1e20)] = 0
    flow_map = flow_map.astype(int)
    start_time = time.time()
    total_set = set()
    sets = {}
    set_ind = 0
    rolls = [(1 - np.floor(i / 3).astype(int), i % 3 - 1) for i in range(9)]
    print(rolls)
    for lat in range(len(flow_map)):
        for lon in range(len(flow_map[0,:])):
            if flow_map[lat,lon] != 5 and flow_map[lat,lon] != 0:
                if (lat,lon) not in total_set:
                    sets[set_ind] = set()
                    end = False
                    while not end:
                        sets[set_ind].add((lat,lon))
                        if flow_map[lat,lon] == 5:
                            end = True
                        elif flow_map[lat,lon] == 0:
                            print("Error! reached a zero")
                            end = True
                        la = lat
                        lo = lon
                        lat += rolls[flow_map[la,lo]-1][0]
                        lon += rolls[flow_map[la,lo]-1][1]

                    total_set.update(sets[set_ind])
                    set_ind += 1

    print(len(sets))


    final_sets = {}
    for i in range(len(sets.keys())):
        if type(sets[i]) != bool:
            for j in range(len(sets.keys())):
                if type(sets[j]) != bool and type(sets[i]) != bool and i != j:
                    if not sets[i].isdisjoint(sets[j]):
                        sets[i].update(sets[j])
                        sets[j] = False

    final_ind = 0
    for i in range(len(sets.keys())):
        if type(sets[i]) != bool:
            final_sets[final_ind] = sets[i]
            final_ind += 1
    print(len(final_sets))
    with open("D:\\basins_test.pkl", "wb") as f:
        pickle.dump(final_sets,f)
    end_time = time.time()
    print(f"This took {np.round(end_time - start_time, 3)} seconds")

    with open("D:\\basins_test.pkl", "rb") as f:
        final_sets = pickle.load(f)

    array_inds = {}
    masks = np.ones((len(array_inds), 2160, 4320))
    for k in range(len(final_sets.keys())):
        array_inds[k] = tuple([[final_sets[k][i][0] for i in range(len(final_sets[k]))],
                               [final_sets[k][i][1] for i in range(len(final_sets[k]))]])
        masks[k,array_inds[k]] = 0

    hello = MPGTModel(directory + filename, "nizetto", None)
    out_file = "D:\\basins_set_test.nc"
    toinclude = ["lat", "lon"]
    dataset = hello.copy_netCDF_vars(hello.river_chars, out_file, toinclude)

path = r"D:\\WWTPs data.xlsx"
training_data = pd.read_excel(path, sheet_name="washing machine data", nrows=45)
df_out = pd.read_excel(path, sheet_name="Household number per country")

hdis = np.array(df_out["HDI"])[:,np.newaxis]
tx = np.array(training_data["HDI"])[:,np.newaxis]
#print(tx)
ty = np.array(training_data["Ownership rate"])[:,np.newaxis]
# print(training_data["Ownership rate"].iloc[0])
# print(training_data)
# print(tx)

# print(ty)
weight_options = ["distance", "uniform"]
for weights in weight_options:
    # fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10,12), sharey='all', sharex='all' )
    fig, axs = plt.subplots(figsize=(6, 5)) #, sharey='all', sharex='all')
    inds = [(0,0), (0,1), (0,2),(1,0), (1,1), (1,2),(2,0), (2,1), (2,2),(3,0), (3,1), (3,2)]
    print(hdis)
    for i in range(len(inds)):
        out = KNeighborsRegressor(n_neighbors=i+1,weights=weights,
                                  algorithm='ball_tree'
                                  ).fit(tx,ty).predict(hdis)
        df_out["Ownership rate NN"] = out
        # #print(new)
        # if i == 4 and weights == 'uniform':
        #     with pd.ExcelWriter(path, mode="a") as writer:
        #         df_out.to_excel(writer, sheet_name="final_data")

        #fig = plt.figure(figsize=(8,8))
        if i==4:
            axs.plot(df_out["HDI"].loc[df_out["HDI"] != 0], df_out["Ownership rate NN"].loc[df_out["HDI"] != 0],
                              linewidth = 0,
                              marker = 'o',
                              label = f"K = {i+1}", markersize=4, color="cornflowerblue")
            #plt.plot(df_out["HDI"], out_2, linewidth = 0, marker = "s", label= "K = 3")
            axs.plot(tx,ty, linewidth = 0, marker = 'x', label="Data", markersize=4, color="orange")
            axs.set_xlabel("HDI")
        #
        # if i%3 == 0:
            axs.set_ylabel("Washing machine ownership rate")
            axs.legend()
    plt.tight_layout()
    plt.savefig(f"D:\\final_outputs\\nearest_neighbor_washingmachines_{weights}.png", dpi=100)
    plt.show()