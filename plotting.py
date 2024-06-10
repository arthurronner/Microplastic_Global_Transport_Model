import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#from netcdf_test import
def plot_lat_lon_map(self):
    test_mat = np.array([[5, 5, 4, 1, 5, 5],
                         [5, 7, 4, 2, 1, 5],
                         [5, 5, 6, 8, 3, 5],
                         [5, 6, 7, 9, 1, 5],
                         [5, 7, 3, 9, 4, 5],
                         [5, 7, 4, 2, 1, 5],
                         [5, 5, 6, 8, 3, 5],
                         [5, 6, 7, 9, 1, 5],
                         [5, 7, 3, 9, 4, 5],
                         [5, 5, 6, 6, 5, 5]]).T
    print(test_mat.shape)
    lat = np.array([0, 1, 2, 3, 4, 5])[::-1]
    lon = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    self.calculate_cell_length(lat, lon, test_mat, test=True)
    arr = [test_mat, np.round(self.dists / 100000, 2)]
    fig, axs = plt.subplots(nrows=2, figsize=(5, 10))

    labels = ["input values", "flow map", "output values"]
    for i in range(len(axs)):
        im = axs[i].imshow(arr[i], origin="lower")
        for j in range(len(test_mat[:, 0])):
            for k in range(len(test_mat[0, :])):
                var = arr[i][j, k]
                text = axs[i].text(k, j, var,
                                   ha="center", va="center", color="w")
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(labels[i])
        # plt.colorbar(im,ax=axs[i])
    plt.tight_layout()
    plt.show()


def plot_flow_map(self, calc_flow=False):
    test_mat = np.array([[5, 5, 5, 5, 5, 5],
                         [5, 7, 4, 2, 1, 5],
                         [5, 5, 6, 8, 3, 5],
                         [5, 7, 3, 9, 4, 5],
                         [5, 5, 5, 5, 5, 5]])
    x, y = self.flow_dir_relation(test_mat)
    value_array = np.linspace(0, len(test_mat) * len(test_mat[0, :]) - 1, len(test_mat) * len(test_mat[0, :])).astype(
        int)
    value_array.shape = test_mat.shape
    arr = [value_array, test_mat, value_array[y, x]]
    if calc_flow == True:
        out_arr = np.copy(value_array)
        flow = np.copy(value_array)
        out_arr = self.discharge_cells(value_array, flow, test_mat)
        arr = [value_array, test_mat, out_arr]
        # print(np.sum(out_arr))
        # print(np.sum(value_array))
    fig, axs = plt.subplots(ncols=3)

    labels = ["input values", "flow map", "output values"]
    for i in range(len(axs)):
        axs[i].imshow(arr[i].T, origin="lower")
        for j in range(len(test_mat[:, 0])):
            for k in range(len(test_mat[0, :])):
                var = arr[i][j, k]
                text = axs[i].text(j, k, var,
                                   ha="center", va="center", color="w")
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(labels[i])
    plt.tight_layout()
    text = ""
    if calc_flow:
        text = "_2"
    plt.savefig(f"flowmap_example{text}.png", dpi=150)
    plt.show()


import seaborn as sns
def wwtp_GNI_plot():
    df = pd.read_excel("D:\\WWTPs data.xlsx", sheet_name="total_reduced", usecols="A:I")
    markers = {
        "Primary": "o",
        "Secondary": "^",
        "Tertiary": "s"
    }
    sns.set_theme(rc={'figure.figsize':(8,8)})
    ax = sns.scatterplot(df,x="GNI index", y="effluent in numbers", hue="Location", size="WWTP number",
                         style="Treatment processes", markers=markers, legend="brief", sizes=(25,400))
    start = True
    for ind, row in df.iterrows():
        if row["WWTP number"] > 1:

            if start:
                low_y = row["effluent in numbers"]
                print(low_y)
                start = False
            else:
                print("making a line")
                end_y = row["effluent in numbers"]
                plt.vlines(row["GNI index"], low_y, end_y, linewidth=0.75,color="black")
                start=True
        #ax.text(row["GNI index"] -100, row["effluent in numbers"], row["Location"], horizontalalignment='right')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Microplastic emissions (P/L)")
    ax.set_xlabel("GNI index ($)")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("MP output vs GNI index logscale_tot.png", dpi=150)
    plt.show()


def wwtp_discharge_plot():
    df = pd.read_excel("D:\\WWTPs data.xlsx", sheet_name="Sun_reduced", usecols="A:H")
    markers = {
        "Primary": "o",
        "Secondary": "^",
        "Tertiary": "s"
    }
    sns.set_theme(rc={'figure.figsize': (8, 8)})
    ax = sns.scatterplot(df, x="Discharge", y="effluent in numbers", hue="Location", size="WWTP number",
                         style="Treatment processes", markers=markers, legend="brief", sizes=(100, 400))
    start = True
    for ind, row in df.iterrows():
        if row["WWTP number"] > 1:

            if start:
                low_y = row["effluent in numbers"]
                low_x = row["Discharge"]
                print(low_y)
                start = False
            else:
                print("making a line")
                end_y = row["effluent in numbers"]
                end_x = row["Discharge"]
                plt.plot([low_x, end_x], [low_y, end_y], linewidth=0.75, color="black")
                start = True
        # ax.text(row["GNI index"] -100, row["effluent in numbers"], row["Location"], horizontalalignment='right')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Microplastic emissions (P/L)")
    ax.set_xlabel("Discharge (m3/year)")
    ax.set_yscale("log")
    #ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig("MP output vs discharge 2.png", dpi=150)
    plt.show()

def plot_wwtp_outputs_old(file="D:\\point_sources_excel_full.xlsx", est_names = ["volgare", "galvao"], calculate = False):
    df = pd.read_excel(file)
    #ToDo: Make shure that this works for more categories :)
    #Or make the image better/ less subfigures
    #Maybe copy the NS subplotting methodology
    if calculate:
        df["abs_difference"] = np.abs(df[f"washing_based_{est_names[1]}"] - df[f"washing_based_{est_names[0]}"])
        df["rel_difference"] = df[f"washing_based_{est_names[0]}"]/df[f"washing_based_{est_names[1]}"]
        df[f"c_washing_{est_names[1]}"] = df[f"washing_based_{est_names[1]}"]/(1000*df["WASTE_DIS"])
        df[f"c_washing_{est_names[0]}"] = df[f"washing_based_{est_names[0]}"]/(1000*df["WASTE_DIS"])
        df["pop_dis_ratio"] = df["POP_SERVED"]/df["WASTE_DIS"]
        df.to_excel(file)
    cols = [[f"washing_based_{est_names[0]}", f"washing_based_{est_names[1]}"],
            [f"c_washing_{est_names[0]}", f"c_washing_{est_names[1]}"],
            "rel_difference",
            "POP_SERVED",
            "pop_dis_ratio"
            ]
    titles = ["Discharge (P/d)", "Effluent concentration (P/L)", "Estimation difference",
              "Population served", "Population discharge ratio"]
    fig, axs = plt.subplots(ncols=len(cols), nrows=1, figsize=(13,7))
    for i in range(len(axs)):
        ax = df.boxplot(column=cols[i], ax=axs[i], showfliers=True)
        #axs[i].violinplot(dataset=df[cols[i]].values)
        #ax.set_title(cols[i])
        ax.set_title(titles[i])
        ax.set_yscale("log")
    # ax = df.boxplot(column=cols,showfliers=False,figsize=(5,10))
    # ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("D:\\input_test_.png", dpi = 150)
    plt.show()

def plot_wwtp_estimates(file="D:\\point_sources_excel_full.xlsx",
                        cols = ["washing_based_galvao", "washing_based_belzagui"],
                        labels = ["Galvao", "Belzagui"],
                        scale_log = True, show_fliers=True, calculate = True, show = False, outfile="D:\\test.png",
                        outliers = True, treat_type = None):

    df = pd.read_excel(file)
    if calculate:
        for i in range(len(cols)):
            df[f"p_L_{labels[i]}"] = df[f"{cols[i]}"]/(1000*df["WASTE_DIS"])
        df.to_excel(file)
    if not outliers:
        df = df[(df["WASTE_DIS"] > 1) & (df["POP_SERVED"] > 1)]

    treat_text = " all treatment"
    if treat_type:
        df = df[df["LEVEL"] == treat_type]
        treat_text = " " + treat_type + " treatment"

    fig, axs = plt.subplots(ncols=2, figsize=(12,9))
    fig.suptitle(f"Wastewater treatment plants discharge estimates{treat_text}")
    df.boxplot(column=cols, ax=axs[0], showfliers=show_fliers)
    axs[0].set_title("Daily discharge P/d")
    df.boxplot(column=[f"p_L_{labels[i]}" for i in range(len(cols))], ax=axs[1], showfliers=show_fliers)
    axs[1].set_title("MP concentration P/L")
    if scale_log:
        axs[0].set_yscale("log")
        axs[0].set_ylim(bottom=6e2, top=3e12)
        axs[1].set_yscale("log")
        axs[1].set_ylim(bottom=1e-3, top=6e6)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    if show:
        plt.show()

# estimates = ["azizi_direct", "washing_azizi_low" , "washing_azizi_belzagui", "washing_based_belzagui"
#              , "washing_based_volgare", "washing_based_galvao"] #, "washing_based_belzagui"]
# names = ["Azizi_direct", "Azizi_low", "Azizi_B", "Belzagui", "Volgare", "Galvao"] #, "Belzagui"]
estimates = ["washing_lower", "washing_avg", "washing_upper"]
names = ["Lower", "Average", "Upper"]
# plot_wwtp_outputs_old(est_names=estimates, calculate = True)
treats = [None, "Primary", "Secondary", "Advanced"]
res = lambda x: x if x else "all"
calc = lambda x: False if x else True
for i in treats:
    plot_wwtp_estimates(file="D:\\point_sources_10april.xlsx", cols = estimates, labels=names, show=True,
                        calculate=calc(i),
                        show_fliers=True, scale_log=True, outliers=False, treat_type=i,
                        outfile=f"D:\\out_tests\\{res(i)}_estimates_10april_na.png")

