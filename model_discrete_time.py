import pandas as pd
import netCDF4 as nc
import h5py  # Need to import this library to account for a dependency error
import numpy as np
import time
#from cftime import date2num, num2date


def print_time_diff(t_start, label="calc"):
    t_end = time.time()
    print(f"calculation {label} took {np.round(t_end - t_start, 3)} seconds")
    return t_end


def write_model_description(dataset,
                            author="Arthur Ronner",
                            institution="Leiden University, TU Delft",
                            title="Microplastic global transport model output",
                            history=None):
    dataset.author = author
    dataset.institution = institution
    dataset.title = title
    if history is None:
        dataset.history = "Created" + time.ctime(time.time())


def copy_netCDF_vars(input_dataset, output_file, to_include, copy_all_time_values=True):
    """Helper function to copy the existing dimension structure from an existing netCDF input file."""
    dataset = nc.Dataset(output_file, "w", format="NETCDF4")
    for name, dimension in input_dataset.dimensions.items():
        dataset.createDimension(
            name, (len(dimension) if (not dimension.isunlimited() and name != "time") else None))

    for name, variable in input_dataset.variables.items():
        if name in to_include:

            x = dataset.createVariable(name, variable.datatype, variable.dimensions)
            dataset[name].setncatts(input_dataset[name].__dict__)
            if not copy_all_time_values and name == "time":
                continue

            dataset[name][:] = input_dataset[name][:]
            # copy variable attributes all at once via dictionary
    return dataset


def initialize_datasets_and_vars(obj, river_char_file=None, Q_in_file=None, T_in_file=None,
                                 emission_in_file=None, dt=1):
    """Helper function to initialise variables that are used by both the data preparation module and the actual model.
    """
    # ToDo: This function is useless. Should just put it back in the init functions for data prep and model
    if emission_in_file:
        obj.emissions = nc.Dataset(emission_in_file, mode="r")
    if Q_in_file:
        obj.river_Q_over_time = nc.Dataset(Q_in_file, mode="r")
    if T_in_file:
        obj.river_T_over_time = nc.Dataset(T_in_file, mode='r')
    if river_char_file:
        obj.river_chars = nc.Dataset(river_char_file, mode='r')

    obj.rolls = [(1 - np.floor(i / 3).astype(int), i % 3 - 1) for i in range(9)]
    obj.dist_interval = np.float32(5 / 60)
    obj.num_lat = int(180 / obj.dist_interval) + 1
    obj.num_lon = int(360 / obj.dist_interval) + 1

    obj.dt = dt


class DataPreparationModule:
    """This module contains all functions related to preparing the input data for the model. All of these only have
    to be executed once."""

    # ToDo: Maybe just make all of the functions here static?
    # They mostly only need part of the inputs and are pretty seperated from each other I feel like.
    def __init__(self, river_in_file=None, Q_in_file=None):
        # These will all be initialised by the function initialize_datasets_and_vars
        self.river_Q_over_time = None
        self.river_T_over_time = None
        self.river_chars = None
        self.dt = 1
        self.num_lat = None
        self.num_lon = None
        self.emissions = None
        self.dist_interval = None
        self.rolls = None

        initialize_datasets_and_vars(self, river_char_file=river_in_file, Q_in_file=Q_in_file)
        self.mp_per_wash = {
            "galvao": 8102855,  # Based on calculations from Galvaõ (2020) (and their lin. relation)
            "volgare": 2656080,  # Based on the work of Volgare et al. (2021)
            "belzagui": 976109,  # Based on the work of Belzagui et al. (2019), this is their mean value
            "low": 464814,  # Lowest value of Belzagui
            "high": 1487405  # Highest value of Belzagui
        }  # and 68% of a 6kg load
        self.mp_fiber_fraction = 0.527  # Based on Sun et al. (2019)
        self.ratio_handwash = 1853 / 23723  # Based on research from Wang et al. (2023)

    def wwtp_washing_est_parametric(self, i, point_sources, est_type, inputs=None, washing_info="D:\\WWTPs data.xlsx",
                                    **kwargs):
        efficiencies = {
            "Primary": inputs["wwtps_rem_eff_primary"],
            "Secondary": inputs["wwtps_rem_eff_secondary"],
            "Advanced": inputs["wwtps_rem_eff_advanced"]
        }

        eff = 1 - efficiencies[point_sources["LEVEL"].iloc[i]]
        if not hasattr(self, "wash_number"):
            self.wash_number = pd.read_excel(washing_info, sheet_name="final_data")

        country_id = self.wash_number["ISO Code"] == point_sources["CNTRY_ISO"].iloc[i]
        # print(f'Country code: {point_sources["CNTRY_ISO"].iloc[i]} index: {i} ')

        wm_ratio = self.wash_number["Ownership rate NN"].loc[country_id].iloc[0]

        x = point_sources["POP_SERVED"].iloc[i].astype(np.float32) * \
            self.wash_number["Wash number"].loc[country_id].iloc[0] \
            / (self.wash_number["Average household size (number of members)"].loc[country_id].iloc[0] *
               365 * self.dt) * \
            (
                    inputs["wwtps_washing_mp"] * wm_ratio + inputs["wwtps_washing_mp"] *
                    self.ratio_handwash * (1 - wm_ratio)
            ) * eff / self.mp_fiber_fraction
        return np.float32(x)

    def wwtp_washing_estimate(self, i, point_sources, est_type, **kwargs):
        selected = False
        if 'lower' in est_type:
            self.mp_fiber_fraction = 0.68
            key = 'low'
            efficiencies = {
                "Primary": 0.881,  # removal efficiency as calculated by Azizi et al. (2022)
                "Secondary": 0.966,  #
                "Advanced": 0.9955  #
            }  # Based on Azizi et al. (2022)
            selected = True
        elif 'upper' in est_type:  # Highest emission estimate
            self.mp_fiber_fraction = 0.32
            key = 'high'
            efficiencies = {
                "Primary": 0.766,  # removal efficiency as calculated by Azizi et al. (2022)
                "Secondary": 0.938,  #
                "Advanced": 0.987  #
            }  # Based on Azizi et al. (2022)
            selected = True
        elif 'avg' in est_type:
            self.mp_fiber_fraction = 0.527
            key = 'belzagui'
            efficiencies = {
                "Primary": 0.833,  # removal efficiency as calculated by Azizi et al. (2022)
                "Secondary": 0.955,  #
                "Advanced": 0.9922  #
            }  # Based on Azizi et al. (2022)
            selected = True
        elif "azizi" in est_type:
            efficiencies = {
                "Primary": 1 - 20.67 / 124.04,  # removal efficiency of 83% as calculated by Azizi et al. (2022)
                "Secondary": 1 - 5.62 / 124.04,  # 95.5%
                "Advanced": 1 - 0.96 / 124.04  # 99.2%
            }  # Based on Azizi et al. (2022)

        else:
            efficiencies = {
                "Primary": 0.72,
                "Secondary": 0.88,
                "Advanced": 0.94
            }  # Based on Iyare et al. (2020)
        if not selected:
            if "galvao" in est_type:
                key = "galvao"
            elif "volgare" in est_type:
                key = "volgare"
            elif "belzagui" in est_type:
                key = "belzagui"
            elif "low" in est_type:
                key = 'low'
            else:
                key = "galvao"
        eff = 1 - efficiencies[point_sources["LEVEL"].iloc[i]]
        if not hasattr(self, "wash_number"):
            self.wash_number = pd.read_excel("D:\\WWTPs data.xlsx", sheet_name="final_data")

        country_id = self.wash_number["ISO Code"] == point_sources["CNTRY_ISO"].iloc[i]
        # print(f'Country code: {point_sources["CNTRY_ISO"].iloc[i]} index: {i} ')

        wm_ratio = self.wash_number["Ownership rate NN"].loc[country_id].iloc[0]

        x = point_sources["POP_SERVED"].iloc[i].astype(np.float32) * \
            self.wash_number["Wash number"].loc[country_id].iloc[0] \
            / (self.wash_number["Average household size (number of members)"].loc[country_id].iloc[0] * 365 * self.dt) * \
            (
                    self.mp_per_wash[key] * wm_ratio + self.mp_per_wash[key] * self.ratio_handwash * (1 - wm_ratio)
            ) * eff / self.mp_fiber_fraction
        return np.float32(x)

    def wwtp_emissions_azizi_direct(self, i, point_sources, *args, **kwargs):
        emissions_per_m3 = {
            "Primary": 20.67 * 1000,
            "Secondary": 5.62 * 1000,
            "Advanced": 0.96 * 1000
        }
        return np.float32(emissions_per_m3[point_sources["LEVEL"].iloc[i]] * point_sources["WASTE_DIS"].iloc[i])

    @staticmethod
    def wwtp_population_served(i, point_sources, *args, **kwargs):
        return point_sources["POP_SERVED"].iloc[i].astype(np.float32)

    @staticmethod
    def wwtp_pop_estimate(i, point_sources, *args, **kwargs):
        efficiencies = {
            "Primary": 0.72,
            "Secondary": 0.88,
            "Advanced": 0.94
        }
        eff = 1 - efficiencies[point_sources["LEVEL"].iloc[i]]
        return 10 ** (3.01 + 1.01 * np.log(point_sources["POP_SERVED"].iloc[i].astype(np.float32))) * eff

    def point_source_emissions_to_netCDF(self, point_sources_file, out_file, out_excel="D:\\point_sources_extra.xlsx",
                                         saved_vars=[None], funcs=None, inputs=None, save_excel='all',
                                         washing_file="D:\\WWTPS data.xlsx"):
        """Translate an input CSV file containing point source emissions (for now, WWTPs) to a netCDF file that we
        can work with."""
        # ToDo: it might make more sense to remove this function from the main class at some point, it does not actually
        # Contribute to the actual modelling
        toinclude = ["lat", "lon"]
        dataset = copy_netCDF_vars(self.river_chars, out_file, toinclude)
        point_sources = pd.read_csv(point_sources_file, encoding="latin-1")

        lat_grid = np.array(np.floor((90 - point_sources["LAT_OUT"]) / self.dist_interval).astype(int))
        lon_grid = np.array(np.floor((point_sources["LON_OUT"] + 180) / self.dist_interval).astype(int))

        if funcs is None:
            funcs = [self.wwtp_washing_est_parametric]

        # WWTPs on the same squares and save the right total data. Will have to see if this is actually needed.
        # saved_vars = ["azizi_direct", "washing_azizi_belzagui", "washing_azizi_low", "washing_based_belzagui",
        #               "washing_based_galvao", "washing_based_volgare", "washing_upper", "washing_lower", "washing_avg"]
        # funcs = [self.wwtp_emissions_azizi_direct, self.wwtp_washing_estimate, self.wwtp_washing_estimate,
        #          self.wwtp_washing_estimate, self.wwtp_washing_estimate, self.wwtp_washing_estimate,
        #          self.wwtp_washing_estimate, self.wwtp_washing_estimate, self.wwtp_washing_estimate]

        for var in range(len(saved_vars)):
            dataset.createVariable(saved_vars[var], "f4", ("lat", "lon"), fill_value=0)
            value_list = np.zeros(len(point_sources["LAT_OUT"]))
            for i in range(len(point_sources["LAT_OUT"])):
                val = funcs[var](i, point_sources, saved_vars[var], inputs=inputs, washing_info=washing_file)
                value_list[i] = val
                # print(set(point_sources["CNTRY_ISO"].unique())-set(self.wash_number["ISO Code"].unique()))
                if type(dataset[saved_vars[var]][lat_grid[i], lon_grid[i]]) == np.ma.core.MaskedConstant:
                    dataset[saved_vars[var]][lat_grid[i], lon_grid[i]] = val
                else:
                    dataset[saved_vars[var]][lat_grid[i], lon_grid[i]] += val

            point_sources[saved_vars[var]] = value_list
        # dataset.createVariable("rel_difference","f4", ("lat", "lon"), fill_value = 0)
        # dataset["rel_difference"][:,:] = dataset["washing_based_volgare"][:,:]/dataset["washing_based_galvao"][:,:]
        dataset.close()
        if save_excel == 'all':
            point_sources.to_excel(out_excel)
        elif save_excel == 'data_only':
            point_sources[saved_vars].to_excel(out_excel)

    def calculate_cell_length(self, lat, lon, flow_dir_array, test=False):
        import geopy.distance as di
        self.dists = np.zeros_like(flow_dir_array)
        mask = np.ma.getmaskarray(flow_dir_array)
        self.dists = np.ma.masked_array(self.dists, mask=mask)
        loopval = int(360 / self.dist_interval)
        if test:
            loopval = len(lat)
        for y in range(len(lat)):
            for x in range(len(lon)):
                if not mask[y, x]:
                    # print(flow_dir_array[y,x])
                    if int(flow_dir_array[y, x]) != 5:
                        roll = self.rolls[int(flow_dir_array[y, x]) - 1]
                        self.dists[y, x] = di.distance((lat[y], lon[x]), (lat[y + roll[1]],
                                                                          lon[(x + roll[0]) % loopval])).m
                    else:
                        self.dists[y, x] = di.distance((lat[y], lon[x]),
                                                       ((90 + lat[y] + self.dist_interval / 2) % 180 - 90,
                                                        lon[x] + self.dist_interval / 2)).m

    def save_distances(self, filename="test.nc", new_file=True):
        self.calculate_cell_length(self.river_chars["lat"][:], self.river_chars["lon"][:],
                                   self.river_chars["lddMap"][:, :])
        if new_file:
            toinclude = ["lat", "lon"]
            dataset = copy_netCDF_vars(self.river_chars, filename, toinclude)
        else:
            self.river_chars.close()
            dataset = nc.Dataset(filename, "a", format="NETCDF4")
        dataset.createVariable("river_lengths", "f4", ("lat", "lon"), fill_value=0)
        dataset["river_lengths"][:, :] = self.dists
        dataset.close()
        self.river_chars = nc.Dataset(filename, "r")

    def calculate_traversing_speeds(self, tot_tstep=None, output_file="trav_speeds.xlsx", return_array=False,
                                    Q_min=1, cs_min=2, return_full_array=False):
        start_time = time.time()
        cross_sections = self.river_chars["bankfull_width"][:, :] * self.river_chars["bankfull_depth"][:, :]
        if tot_tstep:
            num_input_data = tot_tstep
        else:
            num_input_data = len(self.river_Q_over_time["time"])
        agg_trav_times = np.zeros((num_input_data, 4))
        for i in range(num_input_data):
            speeds = self.river_Q_over_time["discharge"][i, :, :] / cross_sections
            trav_times = self.river_chars["river_lengths"] / speeds
            selection = np.where((self.river_Q_over_time["discharge"][i, :, :] > Q_min) & (cross_sections > cs_min))
            if return_full_array:
                return trav_times[selection]
            agg_trav_times[i, 0] = np.mean(trav_times[selection])
            agg_trav_times[i, 1] = np.std(trav_times[selection])
            agg_trav_times[i, 2] = np.ma.min(trav_times[selection])
            agg_trav_times[i, 3] = np.ma.max(trav_times[selection])
            if i % 10 == 0 and i != 0:
                end_time = time.time()
                print(f"Done with timestep {i}, took {np.round(end_time - start_time, 3)} seconds")
                start_time = time.time()
        df = pd.DataFrame(agg_trav_times)
        df.to_excel(output_file, index=False)
        if return_array:
            return agg_trav_times

    @staticmethod
    def mp_shape_dist(arr,
                      f1=0.06, f2=0.94, o1=0.03, o2=0.19, mu1=0.08, mu2=0.44):

        y = f1 * 1 / ((2 * np.pi * o1 ** 2) ** 0.5) * np.exp(-(arr - mu1) ** 2 / (2 * o1 ** 2)) + \
            f2 * 1 / ((2 * np.pi * o2 ** 2) ** 0.5) * np.exp(-(arr - mu2) ** 2 / (2 * o2 ** 2))
        return y

    @staticmethod
    def mp_size_dist(arr, xmin=20, xmax=5000, a=1.6):
        b = (a - 1) * xmin ** (a - 1)
        y = b * (arr * 1e6) ** (- a)
        return y

    @staticmethod
    def mp_density_dist(arr,
                        mu=0.84, d=0.097, a=75.1, b=71.3):
        from scipy.special import kn

        y = a * d * kn(1, a * (d ** 2 + (arr / 1000 - mu) ** 2) ** 0.5) / \
            (np.pi * (d ** 2 + (arr - mu) ** 2)) ** 0.5 * np.exp(d * (a ** 2 + b ** 2) ** 0.5 + b * (arr / 1000 - mu))

        return y


    def create_mp_categories(self, input_file, type_factors=None, print_occ=False):
        """Helper function to create microplastic categories from an input excell file. Translates into a dictionary
        that can be used by a MPGTModel class instance.

        Each property will be an ordered numpy array."""
        df = pd.read_excel(input_file, sheet_name="mp_data")
        cols = df.to_dict(orient="list")
        for i in cols.keys():
            cols[i] = np.array(cols[i])
        cols['density_occurrence'] = self.mp_density_dist(cols["density"])
        cols['size_occurrence'] = self.mp_size_dist(cols["a"])
        cols['shape_occurrence'] = self.mp_shape_dist(cols["CSF"])
        if type_factors is None:
            cols['occurrence'] = cols['density_occurrence'] * cols['size_occurrence'] * cols['shape_occurrence'] / \
                                 np.sum(cols['density_occurrence'] * cols['size_occurrence'] * cols['shape_occurrence'])
        else:
            df = pd.DataFrame.from_dict(cols)

            # type_factors should be a dictionary where the keys are the MP type names.
            totals = df.copy()
            totals['occurrence'] = df['density_occurrence'] * df['size_occurrence'] * df['shape_occurrence']
            totals = totals.groupby('type')[['occurrence']].transform('sum')
            df['occurrence'] = df['density_occurrence'] * df['size_occurrence'] * df['shape_occurrence'] / \
                                 (totals['occurrence'])
            df['occurrence'] = df.apply(lambda row: row['occurrence'] *
                                                        type_factors[f"wwtps_{row['type']}_fraction"], axis=1)


            columns = df.to_dict(orient="list")
            for i in columns.keys():
                columns[i] = np.array(columns[i])

            cols = columns

        if print_occ:
            print(f"Occurrence of {cols['names']} is")
            print(cols["occurrence"])
            print("percent")
        return cols


class MPGTModel:
    def __init__(self, in_file, flows, mp_properties, load_time_data=True,
                 emission_in_file=None, Q_in_file="", T_in_file="", time_steps=None, time_chunk=10,
                 dt=1, visc_in_file="",
                 save_interval=1,
                 prnt=False,
                 inputs=None,
                 warm_start=None, concat_results=False, emissions_key="washing_azizi_low",
                 print_yu=False,
                 compression_level=-1,
                 partial_run=False,
                 continue_run=None,
                 continue_run_file=''):
        start_time = time.time()
        # These will all be initialised by the function initialize_datasets_and_vars
        self.river_Q_over_time = {}
        self.river_T_over_time = {}
        self.river_chars = {}
        self.dt = 1
        self.num_lat = None
        self.num_lon = None
        self.emissions = None
        self.dist_interval = None
        self.rolls = None
        self.concat_results = concat_results
        self.prnt_yu = print_yu
        self.partial_run = partial_run
        self.compression_level = compression_level
        initialize_datasets_and_vars(self, river_char_file=in_file, Q_in_file=Q_in_file, T_in_file=T_in_file,
                                     emission_in_file=emission_in_file, dt=dt)

        self.dt_seconds = np.float32(self.dt * 3600 * 24)
        self.prnt = prnt
        self.g = np.float32(9.81)  # Worldwide this may vary roughly 0.7% according to wikipedia
        self.water_density = np.float32(997)  # for 25 degrees C, taken as a constant. in kg/m3
        self.K = np.float32(273.15)  # Kelvin to C constant

        if not mp_properties:
            self.mp_density = np.array([1.1], dtype=np.float32)  # made up
            self.D_med = np.array([1.5e-3], dtype=np.float32)  # 1.5 mm, made up
            self.D_low = np.array([1e-3], dtype=np.float32)  # made up
            self.D_upp = np.array([2e-3], dtype=np.float32)  # made up
            self.num_mp_categories = 1
            self.mp_category_names = np.array(["test_cat"])
        else:
            self.mp_density = mp_properties["density"].astype(np.float32)[:, None, None]
            self.D_med = mp_properties["a"].astype(np.float32)[:, None, None]
            self.D_low = mp_properties["alow"].astype(np.float32)[:, None, None]
            self.D_upp = mp_properties["aupp"].astype(np.float32)[:, None, None]
            self.mp_CSF = mp_properties["CSF"].astype(np.float32)[:, None, None]
            self.mp_sphericity = mp_properties["sphericity"].astype(np.float32)[:, None, None]
            self.mp_volume = mp_properties["volume"].astype(np.float32)[:, None, None]
            self.mp_occurrence = mp_properties["occurrence"].astype(np.float32)[:, None, None]
            self.mp_category_names = mp_properties["names"]
            self.num_mp_categories = len(mp_properties["density"])

        self.time_chunk = time_chunk

        # These variables are specific to our entrainment and settling flow, could be moved somewhere later
        # So that hey are not required as an input
        # ToDo: make this test a bit more robust haha
        if inputs is None:
            self.a7 = np.float32((0.08 + 8e-5) / 2)  # middle of ranges
            self.a8 = np.float32((2e-7 + 4e-6) / 2)  # middle or ranges
            self.beta1 = np.float32(-0.25)
            self.beta2 = np.float32(0.03)
            self.beta3 = np.float32(0.33)
            self.beta4 = np.float32(0.25)
            self.model_variables = {'hello': 809, 'text': 1234567}
        else:
            self.a7 = np.float32(inputs["a7"])
            self.a8 = np.float32(inputs["a8"])
            self.beta1 = np.float32(inputs["beta1"])
            self.beta2 = np.float32(inputs["beta2"])
            self.beta3 = np.float32(inputs["beta3"])
            self.beta4 = np.float32(inputs["beta4"])
            if type(inputs) != dict:
                self.model_variables = inputs.to_dict()

        self.visc_table = pd.read_excel(visc_in_file)["viscosity"].values.astype(
            np.float32)  # For temperatures 0 - 50 so 26 indices

        if not time_steps and load_time_data:
            self.time_steps = len(self.river_Q_over_time["time"])
        else:
            self.time_steps = min(time_steps, int((7 / self.dt) * len(self.river_Q_over_time["time"]) ))

        if warm_start is not None:
            #This is a legacy feature that we might want to remove
            self.previous_run = nc.Dataset(warm_start, mode='r')
            self.warm_start = True
            self.start_time = self.previous_run["time"][-1]
            self.elapsed_timesteps = self.previous_run["time"][-1] - self.previous_run["time"][0]
            self.start_timestep = len(self.previous_run["time"])
            self.time_arr = np.floor([(self.elapsed_timesteps + self.dt * i) / 7
                                      for i in range(self.time_steps)]).astype(int)

            self.suspended_mp = np.ma.zeros((self.num_mp_categories,
                                             min(time_steps, self.time_chunk) + 1, self.num_lat, self.num_lon),
                                            dtype=np.float32)
            self.sediment_mp = np.ma.zeros((self.num_mp_categories,
                                            min(time_steps, self.time_chunk) + 1, self.num_lat, self.num_lon),
                                           dtype=np.float32)
            for i in range(len(self.mp_category_names)):
                self.suspended_mp[i, 0, :, :] = self.previous_run[f"sus_mp_{self.mp_category_names[i]}"][-1, :, :]
                self.sediment_mp[i, 0, :, :] = self.previous_run[f"sed_mp_{self.mp_category_names[i]}"][-1, :, :]
            print(self.start_timestep)

        elif continue_run is not None:
            self.start_time = self.river_Q_over_time["time"][0]
            self.warm_start = False
            self.time_arr = np.floor([self.dt * i / 7 for i in range(self.time_steps)]).astype(int)

            self.suspended_mp = np.ma.zeros((self.num_mp_categories,
                                             min(time_steps, self.time_chunk) + 1, self.num_lat, self.num_lon),
                                            dtype=np.float32)
            self.sediment_mp = np.ma.zeros((self.num_mp_categories,
                                            min(time_steps, self.time_chunk) + 1, self.num_lat, self.num_lon),
                                           dtype=np.float32)
            save_dataset = nc.Dataset(continue_run_file,'r')
            print(f'Are the mp names the same? {save_dataset.mp_names} {self.mp_category_names}')
            for i in range(len(self.mp_category_names)):
                self.suspended_mp[i, 0, :, :] = save_dataset[f"sus_mp_{self.mp_category_names[i]}"][continue_run, :, :]
                self.sediment_mp[i, 0, :, :] = save_dataset[f"sed_mp_{self.mp_category_names[i]}"][continue_run, :, :]
            print(np.ma.sum(self.suspended_mp))
            print(np.sum(self.suspended_mp))
            self.start_timestep = continue_run * save_interval + 1
            self.continue_run = continue_run
            save_dataset.close()

        else:
            self.start_time = self.river_Q_over_time["time"][0]
            self.warm_start = False
            self.time_arr = np.floor([self.dt * i / 7 for i in range(self.time_steps)]).astype(int)

            self.suspended_mp = np.ma.zeros((self.num_mp_categories,
                                             min(time_steps, self.time_chunk) + 1, self.num_lat, self.num_lon),
                                            dtype=np.float32)

            self.sediment_mp = np.ma.zeros((self.num_mp_categories,
                                            min(time_steps, self.time_chunk) + 1, self.num_lat, self.num_lon),
                                           dtype=np.float32)
            self.start_timestep = 0
        self.time_offset = int(-6)
        self.saved_timesteps = np.array([self.start_time +
                                         self.time_offset + i for i in range(save_interval, self.dt * self.time_steps,
                                                                            save_interval)])
        self.save_interval = save_interval
        # Here we kind of assume that the save_interval is a multiple of dt!

        # Array for selecting the right Q and T data

        # FOR NOW WE SOLVED THE INITIALISATION BY ADDING ONE MORE TIMESTEP AT THE END
        self.map_mask = np.ma.getmask(self.river_chars["lddMap"][:, :])[np.newaxis, :, :]
        self.suspended_mp.mask = self.map_mask[None, :, :, :]
        self.sediment_mp.mask = self.map_mask[None, :, :, :]

        # Saving a view of the Q and T data, as it is the same each week.
        self.Q_snapshot = self.river_Q_over_time["discharge"][self.time_arr[0], :, :]
        self.T_snapshot = self.river_T_over_time["waterTemperature"][self.time_arr[0], :, :]

        self.initialise_flows(flows, emissions_key)  # ToDo: Parameterise this function
        print(self.suspended_mp.shape)
        end_time = time.time()
        print(f"initialisation took {np.round(end_time - start_time, 3)} seconds")

    def initialise_flows(self, flows, emissions_key):

        if flows == "discharge":
            self.flows = {"discharge": self.river_discharge_flow}
            self.flow_args = {"discharge": {}}
        elif flows == "discharge_emissions":
            self.flows = {
                "discharge": self.river_discharge_flow,
                "emissions": self.emissions_flow
            }
            self.flow_args = {
                "discharge": {},
                "emissions": {"estimate": emissions_key}
            }
        elif flows == "emissions":
            self.flows = {
                "emissions": self.emissions_flow
            }
            self.flow_args = {
                "emissions": {"estimate": emissions_key}
            }
        elif flows == "dis_em_set":
            self.flows = {
                "discharge": self.river_discharge_flow,
                "settling": self.settling_flow,
                "emissions": self.emissions_flow,
            }
            self.flow_args = {
                "discharge": {},
                "settling": {},
                "emissions": {"estimate": emissions_key}
            }
        elif flows == "nizetto":
            self.flows = {
                "discharge": self.river_discharge_flow,
                "emissions": self.emissions_flow,
                "settling": self.settling_flow,
                "entrainment": self.entrainment_flow,
            }

            self.flow_args = {
                "emissions": {"estimate": emissions_key},
                "discharge": {},
                "settling": {},
                "entrainment": {},
            }
        elif flows == "yu":
            self.flows = {
                "discharge": self.river_discharge_flow,
                "emissions": self.emissions_flow,
                "settling": self.settling_flow_Yu,
                "entrainment": self.entrainment_flow,
            }

            self.flow_args = {
                "emissions": {"estimate": emissions_key},
                "discharge": {},
                "settling": {},
                "entrainment": {},
            }

    def entrainment_flow(self, args):

        stream_power = self.water_density * self.g * self.Q_snapshot \
                       / self.river_chars["bankfull_width"][:, :] * self.river_chars["gradient"][:, :]

        r = self.river_chars["bankfull_width"][:, :] * \
            self.river_chars["bankfull_depth"][:, :] / \
            (2 * self.river_chars["bankfull_depth"][:, :] + self.river_chars["bankfull_width"][:, :])

        rmax = self.river_chars["bankfull_width"][:, :] / 4

        friction_factor = r / rmax
        shear_velocity = np.sqrt(args["a7"] * self.g * self.river_chars["bankfull_depth"][:, :]
                                 * self.river_chars["gradient"][:, :])
        D_max = 9.9941 * shear_velocity ** 2.5208
        ent_ratio = (D_max - self.D_low) / (self.D_upp - self.D_low)
        ent_ratio = np.maximum(np.minimum(ent_ratio, 1), 0)
        # We don't devide by L*W as our flow is equal to N/C * L * W = N/ (L * W * D) * L * W = N / D (so it cancels out)
        #
        ent_mass = args["a8"] * (self.sediment_mp[:, self.tstep, :, :]) * stream_power * friction_factor * ent_ratio
        flow = np.maximum(np.minimum(ent_mass *
                                     self.dt_seconds, self.sediment_mp[:, self.tstep + 1, :, :]), 0)

        if self.prnt:
            print(f"entrainment flow size: {np.sum(flow):,}")
            print(f"entrainment dtype: {flow.dtype}")
        # print(f"minimal flow value: {np.min(flow)}")
        self.sediment_mp[:, self.tstep + 1, :, :] -= flow
        self.suspended_mp[:, self.tstep + 1, :, :] += flow

    def river_discharge_flow(self, args):
        # strt_tm = time.time()
        volume = self.river_chars["bankfull_width"][:, :] * self.river_chars["bankfull_depth"][:, :] \
                 * self.river_chars["river_lengths"][:, :]
        # end_tm = time.time()
        # print(f"volume calc took: {np.round(end_tm-strt_tm, 3)} seconds")
        volume.mask = self.map_mask
        Q = self.Q_snapshot * self.suspended_mp[:, self.tstep, :, :] / volume
        Q = np.maximum(np.minimum(Q * self.dt_seconds, self.suspended_mp[:, self.tstep + 1, :, :]), 0)
        Q.mask = self.map_mask
        Q_res = self.discharge_cells(Q, self.river_chars["lddMap"])
        self.suspended_mp[:, self.tstep + 1, :, :] += Q_res
        # self.suspended_mp.mask = self.map_mask
        return

    def discharge_cells(self, flowrate_array, flow_dir_array):
        # outflow_arr = -flowrate_array.copy()
        outflow_arr = np.zeros_like(flowrate_array, dtype=np.float32)
        cells = 0
        view_flow_dir_array = flow_dir_array[:, :]

        for i in range(1, 10):
            mask_arr = np.zeros((self.num_mp_categories, self.num_lat, self.num_lon), dtype=bool)
            mask_arr[np.where((view_flow_dir_array[None, :, :].astype(int) == i) & (flowrate_array != 0))] = 1

            # total_mask += mask_arr
            outflow_arr += np.roll(np.ma.getdata(flowrate_array) * mask_arr, self.rolls[i - 1], axis=(1, 2))
        # print(f"{int(cells)} cells rolled and {outflow_arr.count()} cells total.")
        if self.prnt:
            print(f"discharge net outflow: {np.ma.sum(outflow_arr):,}, gross flow: {np.ma.sum(flowrate_array):,}")
            # print(f"unfiltered outcome: {np.sum(np.ma.getdata(outflow_arr))}, gross: {np.sum(np.ma.getdata(flowrate_array))}")
            print(f"The difference is {np.sum(outflow_arr - flowrate_array):,}")
            print(f"dtypes of outflow {outflow_arr.dtype} and flowrate_array {flowrate_array.dtype}")
        return outflow_arr - flowrate_array

    def emissions_flow(self, args):
        # ToDo: This is the same for each timestep, so in principle we dont have to load this data every time.
        emissions = self.emissions[args["estimate"]][:, :]
        if self.prnt:
            print(f"net emissions: {np.ma.sum(emissions):,}")

        self.suspended_mp[:, self.tstep + 1, :, :] += self.mp_occurrence * np.ma.getdata(emissions)[None, :, :]
        return

    def settling_flow(self, args):
        indices = np.round(((self.T_snapshot - self.K) / 2), 0).astype(
            int)
        indices[np.where(indices < 0)] = 0
        indices[np.where(indices > 25)] = 25
        viscosities = self.visc_table[indices]
        velocity = (self.mp_density - self.water_density) / (18 * viscosities) * self.g * (self.D_med ** 2)
        flow = np.maximum(np.minimum(
            velocity * self.dt_seconds * self.suspended_mp[:, self.tstep, :, :] / self.river_chars["bankfull_depth"],
            self.suspended_mp[:, self.tstep + 1, :, :]
        ), -self.sediment_mp[:, self.tstep + 1, :, :])
        # print(type(flow[0, 0]))
        # print(type(np.ma.getdata(flow)[0, 0]))
        if self.prnt:
            print(f"Net settling flow (Nizetto): {np.sum(np.ma.getdata(flow)):,}")
        self.suspended_mp[:, self.tstep + 1, :, :] -= np.ma.getdata(flow)
        self.sediment_mp[:, self.tstep + 1, :, :] += np.ma.getdata(flow)

        return

    def settling_flow_Yu(self, args):
        t_start = None
        if self.prnt_yu:
            t_start = time.time()

        indices = ((self.T_snapshot - self.K) / 2).astype(
            int)

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="round")

        indices[np.where(indices < 0)] = 0
        indices[np.where(indices > 25)] = 25

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="cutoff")
        viscosities = self.visc_table[indices].astype(np.float32)

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="viscosities")

        d_n = (6 * self.mp_volume / np.pi) ** (1 / 3)

        if self.prnt:
            print(f'dtype of d_n: {d_n.dtype}')
        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="d_n")

        d_star = (self.g * (self.mp_density / self.water_density - 1) / (viscosities ** 2)) ** (1 / 3) * d_n

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="d_start")

        c_ds = (432 / d_star ** 3) * (1 + 0.0022 * d_star ** 3) ** 0.54 + 0.47 * (1 - np.exp(-0.15 * d_star ** 0.45))

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="c_ds")

        c_n = c_ds / (d_star ** self.beta1 * (self.mp_sphericity ** self.beta2) ** d_star
                      * (self.mp_CSF ** self.beta3) ** d_star) ** self.beta4

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="c_n")

        velocity = (viscosities * self.g * (self.mp_density / self.water_density - 1)) ** (1 / 3) * \
                   ((4 * d_star) / (3 * c_n)) ** (1 / 2)

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="velocity")

        flow = np.maximum(np.minimum(
            velocity * self.dt_seconds * self.suspended_mp[:, self.tstep, :, :] / self.river_chars["bankfull_depth"],
            self.suspended_mp[:, self.tstep + 1, :, :]
        ), -self.sediment_mp[:, self.tstep + 1, :, :])

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="flow")
        # print(type(flow[0, 0]))
        # print(type(np.ma.getdata(flow)[0, 0]))
        if self.prnt:
            print(f"Net settling flow (Yu): {np.sum(np.ma.getdata(flow)):,}")
        self.suspended_mp[:, self.tstep + 1, :, :] -= np.ma.getdata(flow)

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="from sus")

        self.sediment_mp[:, self.tstep + 1, :, :] += np.ma.getdata(flow)

        if self.prnt_yu:
            t_start = print_time_diff(t_start, label="to_sed")

    def save_output_file(self, output_file, output, var_name="Cmp", copy_all_time_values=False, mode="w",
                         tdim=0, keep_result_open=False, save_model_vars=False,
                         create_file_description=False):
        """Either pass a dictionary of output arrrays, or an array with output values.
        Save it in the netCDF format to the specified file location."""
        if mode == "w":
            toinclude = ["latitude", "longitude", "time"]
            dataset = copy_netCDF_vars(self.river_Q_over_time, output_file, toinclude,
                                       copy_all_time_values=copy_all_time_values)
            if not copy_all_time_values:
                dataset["time"][:] = self.saved_timesteps

            if type(output) == dict:
                for i in output.keys():
                    dataset.createVariable(i, "f4", ("time", "latitude", "longitude"), fill_value=0, compression='zlib',
                                           least_significant_digit=self.compression_level)
                    dataset[i][tdim, :, :] = np.ma.getdata(output[i])
            else:
                dataset.createVariable(var_name, "f4", ("time", "latitude", "longitude"), fill_value=0)
                dataset[var_name][tdim, :, :] = output

            if save_model_vars:
                dataset.model_var_keys = [*self.model_variables.keys()]
                dataset.model_var_values = [*self.model_variables.values()]
                dataset.mp_names = self.mp_category_names
                #print(self.mp_occurrence[:,0,0])
                dataset.mp_occurrence = self.mp_occurrence[:,0,0]

            if create_file_description:
                write_model_description(dataset)

        elif mode == "a":
            dataset = nc.Dataset(output_file, mode="a")
            if self.warm_start and self.concat_results:
                dataset["time"][tdim] = self.saved_timesteps[tdim - self.start_timestep]
            for i in output.keys():
                # if "sus" in i:
                #     print(np.ma.sum(output[i]))
                #     print(np.sum(np.ma.getdata(output[i])))
                dataset[i][tdim, :, :] = np.ma.getdata(output[i])

        else:
            dataset = nc.Dataset(output_file)

        dataset.close()
        if keep_result_open:
            self.model_outcome = nc.Dataset(output_file, "r")

    def run_model(self, output_file, keep_result_open=False, report_runtime=False,
                  save_model_vars=True, save_file_description=True, close_model=True):
        """Here for now, we use the mass balance method of flow prioritisation."""
        self.output_file = output_file
        start_time = time.time()
        self.tstep = 0

        if not self.warm_start or not self.concat_results:
            first_save = True
            save_ind = 0
        else:
            first_save = False
            save_ind = self.start_timestep

        if self.partial_run:
            first_save = False
            # We assume here that the file we should write to is made OUTSIDE of this model run. Moreover, all
            # things that we do in the first save thing are done there

        # ToDo: We have to make sure that we properly account for initialisation :)
        if "entrainment" in self.flows.keys():
            self.flow_args["entrainment"]["a7"] = self.a7
            self.flow_args["entrainment"]["a8"] = self.a8
        print("running model....")
        if self.start_timestep != 0:
            first_save = False
            save_ind = self.continue_run + 1

        for i in range(self.start_timestep, self.time_steps):
            self.suspended_mp[:, self.tstep + 1, :, :] += np.ma.getdata(self.suspended_mp[:, self.tstep, :, :])
            self.sediment_mp[:, self.tstep + 1, :, :] += np.ma.getdata(self.sediment_mp[:, self.tstep, :, :])
            # First add the old stock to the new timestep, so we can alter it.
            for j in self.flows.keys():
                flow_start = None
                if report_runtime:
                    flow_start = time.time()
                self.flow_args[j]["time"] = self.time_arr[i]  # Grab the right index, as our input data is in weeks
                self.flows[j](self.flow_args[j])

                if report_runtime:
                    flow_end = time.time()
                    print(f"flow {j} took {np.round(flow_end - flow_start, 3)} seconds")
            self.tstep += 1

            if i % self.save_interval == 0 and i != 0:

                if report_runtime:
                    save_start_time = time.time()
                save_dict = {}
                for k in range(self.num_mp_categories):
                    save_dict[f"sus_mp_{self.mp_category_names[k]}"] = self.suspended_mp[k, self.tstep, :, :]
                    save_dict[f"sed_mp_{self.mp_category_names[k]}"] = self.sediment_mp[k, self.tstep, :, :]

                if first_save:
                    self.save_output_file(self.output_file, save_dict,
                                          mode="w", tdim=save_ind,
                                          copy_all_time_values=False, keep_result_open=keep_result_open,
                                          save_model_vars=save_model_vars,
                                          create_file_description=save_file_description)
                    first_save = False
                else:
                    self.save_output_file(self.output_file, save_dict,
                                          mode="a", tdim=save_ind,
                                          keep_result_open=keep_result_open,
                                          save_model_vars=False,
                                          create_file_description=False)
                save_ind += 1

                if report_runtime:
                    save_end_time = time.time()
                    print(f"save time took {np.round(save_end_time - save_start_time, 3)} seconds")
            if self.tstep == self.time_chunk or i == self.time_steps - 1:
                self.tstep = 0

                print(f"we're at timestep {i}")
                if i != self.time_steps - 1:
                    #This could be more efficient by just setting stuff to zero probably.
                    last_sus_mp = self.suspended_mp[:, -1, :, :].copy()
                    last_sed_mp = self.sediment_mp[:, -1, :, :].copy()
                    self.suspended_mp = np.ma.zeros((self.num_mp_categories, min(self.time_steps - i,
                                                                                 self.time_chunk) + 1,
                                                     self.num_lat, self.num_lon), dtype=np.float32)
                    self.suspended_mp.mask = self.map_mask
                    self.sediment_mp = np.ma.zeros((self.num_mp_categories, min(self.time_steps - i,
                                                                                self.time_chunk) + 1,
                                                    self.num_lat, self.num_lon), dtype=np.float32)
                    self.sediment_mp.mask = self.map_mask
                    self.suspended_mp[:, 0, :, :] = last_sus_mp
                    self.sediment_mp[:, 0, :, :] = last_sed_mp
            if i != self.time_steps - 1:
                if self.time_arr[i + 1] != self.time_arr[i]:
                    print("changing Q and T snapshots...")
                    self.Q_snapshot = self.river_Q_over_time["discharge"][self.time_arr[i + 1], :, :]
                    self.T_snapshot = self.river_T_over_time["waterTemperature"][self.time_arr[i + 1], :, :]
            print(f"done with timestep {i}")
        end_time = time.time()
        if close_model:
            self.river_chars.close()
            self.river_Q_over_time.close()
            self.river_T_over_time.close()
        print(f"simulation took {np.round(end_time - start_time, 3)} seconds")


if __name__ == "__main__":
    directory = "C:\\Users\\Arthur\\Documents\\uni bestanden\\Industrial Ecology\\0. master thesis\\model_input\\"
    file_name = "channel_parameters_extended.nc"
    Q_series_file = "D:\\inputs\\discharge_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc"
    T_series_file = "D:\\inputs\\waterTemp_weekAvg_output_E2O_hist_1996-01-07_to_2005-12-30.nc"
    time_steps = 5
    hi = DataPreparationModule()
    mps = hi.create_mp_categories(directory + "mp_categories_test_2.xlsx")
    print(mps["occurrence"])
    times = []
    for comp in [-3]:
        start_t = time.time()

        hello = MPGTModel(directory + file_name, "yu", mps, time_steps=time_steps, Q_in_file=Q_series_file,
                          T_in_file=T_series_file,
                          emission_in_file=directory + "test_wwtps_large.nc", time_chunk=5,
                          save_interval=1,
                          dt=1,
                          visc_in_file=directory + "input_mu.xlsx", compression_level=comp,
                          warm_start=None, concat_results=False, print_yu=False)
        hello.run_model(f"D:\\test_yu_8cat_{comp}_vars.nc", save_file_description=True, save_model_vars=True)
        end_t = time.time()
        times.append(end_t - start_t)

    times = np.round(np.array(times), 3)
    print(times)
    #
    # print(hello.saved_timesteps)
    # print(len(hello.saved_timesteps))
    # print(hello.mp_occurrence
    # hello.calculate_traversing_speeds(output_file="speeds_final.xlsx")

    # hello = DataPreparationModule(river_in_file=directory + file_name)
    # hello.point_source_emissions_to_netCDF(directory + "HydroWASTE_v10.csv", directory + "wwtps_large_10apr.nc",
    #                                       out_excel=
    #                                       "D:\\point_sources_10april.xlsx")
# travel_times = hello.calculate_traversing_speeds(return_full_array=True)
# ax = sns.boxplot(x=travel_times / (24 * 3600))
# ax.set_xlabel("Travel time (d)")
# ax.set_xscale("log")
# plt.savefig("")
# plt.show()
# data = nc.Dataset("D:\\test_run_numba_9.nc", "r")
# print(data)
# hoi = hello.river_T_over_time["waterTemperature"][0,:,:].copy()
# plt.hist(hoi[np.where(hoi <= 273+100)].flatten(), bins=30)
# print(len(hoi[np.where((hoi > 320) & (hoi < 1e10))].flatten()))
# plt.show()
# for i in range(time_steps):
#     print(data["time"][i])
#     print(np.sum(np.ma.getdata(data["sus_mp"][i,:,:])))
#     print(np.ma.sum(data["sus_mp"][i,:,:]))
