from ema_workbench.em_framework.samplers import LHSSampler

from ema_workbench.em_framework.parameters import (
    IntegerParameter,
    BooleanParameter,
    RealParameter,
    CategoricalParameter
    )
import pandas as pd
import numpy as np
from scipy.stats import gamma
from pathlib import Path
import random

def dirichlet_ppf(X, alpha):
    # dirichlet_ppf is not an exact quantile function since the quantile of a
    #  multivariate distribtion is not unique
    # dirichlet_ppf is also not the quantiles of the marginal distributions since
    #  those quantiles do not sum to one
    # dirichlet_ppf is the quantile of the underlying gamma functions, normalized
    # This has been tested to show that dirichlet_ppf approximates the dirichlet
    #  distribution well and creates the correct marginal means and variances
    #  when using a latin hypercube sample
    #
    # Python translation of qdirichlet function by  R. Carnell
    # original: https://stats.stackexchange.com/a/476433/244679

    X = np.asarray(X)
    alpha = np.asarray(alpha)

    assert alpha.ndim == 1, "parameter alpha must be a vector"
    assert X.ndim == 2, "parameter X must be an array with samples as rows and variables as columns"
    assert X.shape[1] == alpha.shape[0], "number of variables in each row of X and length of alpha must be equal"
    assert not (np.any(np.isnan(X)) or np.any(np.isnan(alpha))), "NAN values are not allowed in dirichlet_ppf"

    Y = np.zeros(shape=X.shape)
    for idx, a in enumerate(alpha):
        if a != 0.:
            Y[:, idx] = gamma.ppf(X[:, idx], a)

    return Y / Y.sum(axis=1)[:, np.newaxis]

class samples():
    """
    A class that contains all the information on the inputs of the model runs.

    Most of this information is constructed from the levers, uncertainties
    and policies tabs from the input file.

    For all the attributes that contain file paths, the following holds.
    *The location is relative to the working file when not using airflow.
    When using airflow, this location should include the folder structure
    as set-up in the docker-compose.yaml file.

    Attributes
    ----------
    uncertainties: list of EMA Parameters
        List of all the input uncertainties specified by the user in the
        uncertainties tab of the input.xlsx file. Used to generate samples.

    uncert_names: list of str
        List of all the names of the uncertainties. Used in numerous methods
        to simplify/ improve readability.

    levers: list of EMA Parameters
        List of all the input levers specified by the user in the
        levers tab of the input.xlsx file. Used to generate samples.

    lever_names: list of str
        List of all the names of the uncertainties. Used in numerous methods
        to simplify/ improve readability.

    locations: list of dicts
        List of the ESDL location of all parameters (uncertainties & levers).
        Used to write the samples to the right locations in the esdls.


    policies: pd.DataFrame
        Dataframe that contains the different policies used in the model.
        These are directly read from the policies tab in the input.xlsx file.

    num_policies: int
        The number of policies specified by the user.

    base_json: str
        The location of the base json file to alter for each of the specified
        runs.*

    num_runs_per_scen: int
        The number of runs excecuted per different uncertainty + policy.
        Used to account for stochasticity of a model. Used by the handler
        class to copy the samples created for each different run.

    work_dir: str
        The path to the folder in which the model runs will be executed.*

    base_esdl: str
        The path to the file where the base esdl file to be altered per run
        is located.*

    num_samples: int
        The total amount of created samples.
        Equal to n policies * n (from input file).
        Note that when selecting sobol samplers, this number might change
        after execution, as n could be changed.

    samples: dict
        Dictionary that contains the sampled values for each of the parameters.
        The keys of the dictionary correspond to the names of the parameters.
        The value of each key is its respective sampled values in a np.array.
        These samples are created using the ema workbench.
        The samples are sorted per policy. Hence the values read as:
            pol0 scen0, pol0 scen1, .., pol0 scen x, pol1 scen0, pol1 scen1, ..
            from index 0 -> end.

    input_file: str
        Contains the location of the input file.*
        Should be passed as an argument when creating a samples object.

    Methods
    -------
    __init__(input_file = "inputs.xlsx", n_runs = None,
             initialize_inputs = True, initialize_samples = True)
        Creates the num_samples and input_file attributes.
        Calls the read_input method if initialize_inputs = True.
        Calls the create_samples method if initialize_samples = True.

    read_input(sound=None)
        Reads the data used by the samples class from the specified input file.
        Creates most of the attributes of the samples class.
        Calls read_parameters to create the uncertainties and levers attributes.

    read_parameters()
        Creates the uncertainties and levers attributes from the given input.

    create_samples()
        Creates samples for all uncertainties and levers,
        based on all the inputs constructed when initializing this object.
    """

    def __init__(self, input_file="inputs.xlsx", n_runs=None, initialize_inputs=True,
                 initialize_samples=True, categories_as_str=True, scale_dirichlet=False, used_uncerts=None):
        self.num_samples = n_runs
        self.categories_as_str = categories_as_str
        self.input_file = input_file
        if initialize_inputs:
            self.read_input()

        if initialize_samples:
            self.create_samples()

        if scale_dirichlet:
            if used_uncerts is None:
                used_uncerts = [x for x in self.uncert_names if 'fraction' in x]
            scale_distributions(self, used_uncerts)
        return

    def read_input(self):
        """Function that is called at the creation of a samples object.

        It reads the uncertainties and levers that are listed in the inputs file.
        Calls the read_parameters function to construct the levers
        and uncertainties attributes.
        Creates the locations, policies, and num_policies attributes.
        """

        input_uncert = pd.read_excel(self.input_file, sheet_name="uncertainties")
        input_levers = pd.read_excel(self.input_file, sheet_name="levers")
        self.uncertainties = [None for i in range(len(input_uncert))]
        self.levers = [None for i in range(len(input_levers))]
        self.locations = {}
        self.categorial_pars = []
        self.read_parameters(input_levers, self.levers)
        self.read_parameters(input_uncert, self.uncertainties)

        self.uncert_names = [self.uncertainties[i].name for i in range(len(self.uncertainties))]
        self.lever_names = [self.levers[i].name for i in range(len(self.levers))]

        self.policies = pd.read_excel(self.input_file, sheet_name="policies").to_dict('list')
        self.num_policies = len(self.policies[self.lever_names[0]])

    def read_parameters(self, input_df, write):
        """Helper function that reads and stores the levers and uncertainties.

        It uses the Parameter classes from the EMA workbench.
        It also creates the locations attribute.
        """
        for ind, row in input_df.iterrows():

            if row["type"] == "real":
                try:
                    default = float(row["default"])
                except ValueError:
                    default = None
                write[ind] = RealParameter(row["name"], float(row["lower"]),
                                           float(row["upper"]), default=default)

            elif row["type"] == "int":
                try:
                    default = int(float(row["default"]))
                except ValueError:
                    default = None
                write[ind] = IntegerParameter(row["name"], int(float(row["lower"])),
                                              int(float(row["upper"])), default=default)

            elif row["type"] == "bool":
                if row["default"] != 'nan':
                    if row["default"].lower() == "none":
                        default = None
                    elif row["default"].lower() == "true":
                        default = True
                    elif row["default"].lower() == "false":
                        default = False
                else:
                    default = None
                write[ind] = BooleanParameter(row["name"], default=default)
            elif row["type"] == "cat":
                self.categorial_pars.append(row["name"])
                categories = row["lower"].split(",")
                write[ind] = CategoricalParameter(row["name"], categories=categories, default=row["default"])

            # self.locations[str(row["name"])] = {'unit_type': str(row["unit_type"]),
            #         'unit': str(row["unit"]),'attribute': str(row["attribute"]),
            #         'data': str(row["data"]), 'replace_type': str(row['replace_type'])}

    def create_samples(self):
        """This function creates samples using the EMA workbench.

        It uses the sample sheet in the input file for its settings.
        Note that if Sobol sampler is selected, the number of runs may be altered
        due to sobol needing a specific number of runs to function correctly.
        """

        sample_info = pd.read_excel(self.input_file, sheet_name="sample")

        # self.base_json = sample_info["base_json"].iloc[0]

        self.num_runs_per_scen = sample_info["num_runs_per_scen"].iloc[0]
        # self.work_dir = str(sample_info["work_dir"].iloc[0])
        # self.base_esdl = sample_info["base_esdl"].iloc[0]

        # if len(str(self.base_json)) < 5:
        #     self.base_json = None #this is when the config is invalid.

        if self.num_samples == None:
            number = sample_info["n"].iloc[0]
        else:
            number = self.num_samples

        sam = LHSSampler()
        # Create samples for uncertainties
        uncert_samples = sam.generate_samples(self.uncertainties, number)
        self.num_samples = len(uncert_samples[self.uncert_names[0]])
        pols = {}

        # Merge the different samples for uncertainties with
        # the selected policies
        if self.levers:
            for i in self.lever_names:
                pols[i] = np.repeat(self.policies[i], self.num_samples)

            for i in self.uncert_names:
                uncert_samples[i] = np.tile(uncert_samples[i], self.num_policies)

            combined = {**pols, **uncert_samples}
            self.samples = combined
        else:
            self.samples = uncert_samples
        self.num_samples = len(self.samples[self.uncert_names[0]])

        if self.categories_as_str:

            # Find the parameter in the uncertainties list
            for i in self.categorial_pars:
                param = None
                for par in self.uncertainties:
                    if par.name == i:
                        param = par
                        break
                if not param:
                    continue
                # Loop through all its values and change the index to a string
                self.samples[i] = self.samples[i].astype(str)
                for j in range(len(self.samples[i])):
                    self.samples[i][j] = param.cat_for_index(int(float(self.samples[i][j]))).name
        return


def scale_distributions(sam, keys_to_scale, return_sample=False):

    x = np.zeros((sam.num_samples, len(keys_to_scale)))

    c = 0
    for i in keys_to_scale:
        x[:,c] = sam.samples[i]
        c += 1

    #For now, alpha is not controlable
    alpha_arr = np.ones(len(keys_to_scale))
    scaled = dirichlet_ppf(x, alpha_arr)

    c = 0
    for i in keys_to_scale:
        sam.samples[i] = scaled[:,c]
        c += 1
    #If we are calling the function from outside of the object
    if return_sample:
        return sam

def create_mp_cat_file(input_file, output_file, prnt=False):
    cat = ['fiber', 'fragment', 'foam', 'bead', 'film']
    sample = samples(input_file)
    #print(sample.samples)
    mps = []
    num_mps = len(cat)*sample.num_samples
    for i in range(len(cat)):
        for j in range(sample.num_samples):
            mp = {
                'names': f'{cat[i]}{j+1}',
                'density': sample.samples[f'{cat[i]}_dens'][j],
                'a': sample.samples[f'{cat[i]}_a'][j],
                'type': cat[i]
            }
            mp['alow'] = mp['a'] * 0.9
            mp['aupp'] = mp['a'] * 1.1
            #print(sample.samples[f'{cat[i]}_b'])
            mp['b'] = sample.samples[f'{cat[i]}_b'][j] * mp['a']

            if cat[i] == 'fiber':
                mp['c'] = mp['b']
                mp['CSF'] = mp['c'] / np.sqrt(mp['a'] * mp['b'])
                mp['volume'] = np.pi * (mp['b']/2)**2 * mp['a']
                d_n = (6 * mp['volume'] / np.pi)** (1/3)
                mp['sphericity'] = 4*np.pi* (d_n/2)**2 / ( mp['a'] * np.pi * mp['b'] + 2 * np.pi * (mp['b']/2)**2)

            elif cat[i] == 'fragment' or cat[i] == 'foam' or cat[i] == 'film':
                mp['c'] = sample.samples[f'{cat[i]}_c'][j] * mp['b']

                mp['CSF'] = mp['c'] / np.sqrt(mp['a'] * mp['b'])
                mp['volume'] = mp['a'] * mp['b'] * mp['c']
                d_n = (6 * mp['volume'] / np.pi)** (1/3)
                mp['sphericity'] = 4*np.pi* (d_n/2)**2 / (2 * mp['a'] * mp['b'] + 2 * mp['a'] * mp['c']
                                                         + 2 * mp['b'] * mp['c'])
            elif cat[i] == 'bead':
                mp['c'] = mp['a'] * (sample.samples[f'{cat[i]}_c'][j] *
                                      (sample.samples[f'{cat[i]}_b'][j] - 0.36) + 0.36)
                mp['CSF'] = mp['c'] / np.sqrt(mp['a'] * mp['b'])
                mp['volume'] = 4/3 * np.pi * mp['a'] * mp['b'] * mp['c']
                d_n = (6 * mp['volume'] / np.pi)** (1/3)
                p = 1.6075
                #area of an ellipsoid is approximated like this:
                area = 4 * np.pi * ( (mp['a']**p * mp['b']**p + mp['a']**p * mp['c']**p + mp['b']**p * mp['c']**p)
                                     /3)**(1/p)
                mp['sphericity'] = 4 * np.pi * (d_n / 2) ** 2 / area
            else:
                raise TypeError
            mps.append(mp)

    df = pd.DataFrame.from_records(mps, index=[*range(1,num_mps+1)])
    if prnt:
        print(df)
    df.to_excel(output_file, sheet_name='mp_data')

if __name__ == '__main__':
    #sam=samples("D:\\inputs\\uncert_test.xlsx", scale_dirichlet=True)
    create_mp_cat_file("D:\\inputs\\create_mp_cats.xlsx", "D:\\mp_cats_test.xlsx", prnt=True)
