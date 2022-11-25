import pickle
import traceback
import warnings
from scipy.optimize import curve_fit
import pandas as pd
import numpy
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


# TODO: Code re-usability can be improved by leaps and bounds


class FitToConstants:
    def __init__(self):
        """
        - 0-Arg constructor for the class
        - Set the frame to add Buckingham potential constants
        """
        self.master_raw_data = None
        self.master_results_frame = pd.DataFrame(
            columns=['Element Name', 'Partial Charge', 'Atomic Mass', 'Atomic Radius', 'Atomic Number',
                     'R2', 'A', 'B', 'C', 'Records'])
        self.temp_results_frame = pd.DataFrame(columns=self.master_results_frame.columns)
        self.init_constants = [5000, 0.1, -0.00001]
        self.master_file_uri: str = 'raw_data.xlsx'
        self.threshold_r2 = 0.90
        self.lin_grid_constant = 3
        self.re_eval_init_constants = False
        self.a_lower = 10
        self.a_upper = 10000000
        self.b_lower = 0.001
        self.b_upper = 1.0
        self.c_lower = -1.0
        self.c_upper = 0
        self.a_bounds_de = None
        self.b_bounds_de = None
        self.c_bounds_de = None
        self.a_lower_bound_cf = None
        self.a_upper_bound_cf = None
        self.b_lower_bound_cf = None
        self.b_upper_bound_cf = None
        self.c_lower_bound_cf = None
        self.c_upper_bound_cf = None
        self.set_basic_bounds()
        self.element_state = None
        self.pc_state = None

        with open('element_pc_dict.pkl', 'rb') as f:
            self.element_pc_dict = pickle.load(f)

    # IN-USE
    @staticmethod
    def buckingham_potential(distance, a, b, c):
        """
        - Returns the Buckingham potential for the given values
        :param distance:
        :param a:
        :param b:
        :param c:
        :return:
        """
        warnings.filterwarnings("ignore")
        return a * numpy.exp(-distance / b) - c / numpy.power(distance, 6)

    # IN-USE
    def opt_r2_vals(self, constants, x_vals, y_vals):
        """
        - Used for optimizing function of R2 values evaluated in the differential evaluations
        :param constants: A, B, C values
        :param x_vals: Distance values
        :param y_vals: Energy Values
        :return: R2
        """
        warnings.filterwarnings("ignore")
        y_space = self.buckingham_potential(x_vals, *constants)
        if numpy.all(numpy.isinf(y_space)) or numpy.all(numpy.isnan(y_space)) or numpy.any(numpy.isinf(y_space)):
            y_space[:] = 0
        r2 = r2_score(y_vals, y_space)
        return 1 - r2

    # IN-USE
    def get_master_frame(self):
        """
        - Removes the unnecessary prefixed column (Indexes)
        :return: trimmed working set
        """
        try:
            master_frame = pd.read_excel(self.master_file_uri, sheet_name=0)
            return master_frame.iloc[:, 1:]
        except BaseException:
            traceback.print_exc()

    # IN-USE
    def get_element_pc_frame(self, element=None, partial_charge=None):
        """
        - Retrieves one element and partial charge combination as a frame at a time and saves them as a list
        - The frame is worked on to get the best constants for Buckingham potential
        :param element: Element name if we need to calculate for one Element - PC combination
        :param partial_charge: Partial Charge if we need to calculate for one Element - PC combination
        :return: list of Element-PC frames (One frame if an element and partial charge is specified)
        """
        # Gets the raw data on a dataframe, removes the column with indices
        self.master_raw_data = self.get_master_frame()
        # Commonly used when generating for all the elements
        if element is None and partial_charge is None:
            # Initialise list to store frames
            all_frames = []
            # Iterate through pickled element names
            for ele in self.element_pc_dict:
                # Iterate through partial charges for the given element
                for char_index in range(len(self.element_pc_dict[ele])):
                    charge = float(self.element_pc_dict[ele][char_index])
                    # Appending given Element - PC frame retrieved from master frame to an empty frame that can later be
                    # added to the list of frames we are going to use
                    ind_element_pc_frame = self.master_raw_data.loc[(self.master_raw_data['Element Name'] == ele) &
                                                                    (self.master_raw_data['Partial Charge'] == charge)]
                    # Append frame only if the number of records are more than zero
                    if len(ind_element_pc_frame) != 0:
                        all_frames.append(ind_element_pc_frame)
            return all_frames

        # Used when we need to get all the frames of Element - PC combo for a pre-defined element only
        elif element is not None and partial_charge is None:
            # Initialising an empty array
            element_frames = []
            for char_index in range(len(self.element_pc_dict[element])):
                charge = float(self.element_pc_dict[element][char_index])
                # Appending to the frame of original data
                ind_element_pc_frame = self.master_raw_data.loc[(self.master_raw_data['Element Name'] == element) &
                                                                (self.master_raw_data['Partial Charge'] == charge)]
                # Append frame only if the number of records are more than zero
                if len(ind_element_pc_frame) != 0:
                    element_frames.append(ind_element_pc_frame)
            return element_frames

        # TODO: This use case is when we are trying for just one element (Can be removed or better implemented)
        else:
            element_pc_frame = self.master_raw_data.loc[(self.master_raw_data['Element Name'] == element)
                                                        & (self.master_raw_data['Partial Charge'] == partial_charge)]
            return element_pc_frame

    # IN-USE
    def implement_diff_evolution(self, x_vals, y_vals):
        """
        - Initial A, B, C values are calculated using differential evolution
        - C is hard bound between -0.99 and 0
        :param x_vals: distance vals
        :param y_vals: energy vals
        :return: Constants A, B, C
        """
        parameterBounds = [self.a_bounds_de, self.b_bounds_de, self.c_bounds_de]
        result = differential_evolution(self.opt_r2_vals, parameterBounds, args=(x_vals, y_vals), seed=3)
        return result.x

    # IN-USE
    def set_init_constants(self, x_vals, y_vals):
        print('Running Grid Search for initial constants')
        all_vals = {}
        for init_a in numpy.linspace(self.a_lower_bound_cf, self.a_upper_bound_cf, self.lin_grid_constant):
            for init_b in numpy.linspace(self.b_lower_bound_cf, self.b_upper_bound_cf, self.lin_grid_constant):
                for init_c in numpy.linspace(self.c_lower_bound_cf, self.c_upper_bound_cf, self.lin_grid_constant):
                    print('INITS', init_a, init_b, init_c)
                    opt_constants, _ = curve_fit(self.buckingham_potential, x_vals, y_vals,
                                                 p0=[init_a, init_b, init_c], maxfev=10000,
                                                 bounds=((self.a_lower_bound_cf, self.b_lower_bound_cf,
                                                          self.c_lower_bound_cf),
                                                         (self.a_upper_bound_cf, self.b_upper_bound_cf,
                                                          self.c_upper_bound_cf)))
                    a, b, c = opt_constants
                    y_space = self.buckingham_potential(x_vals, a, b, c)
                    r2 = r2_score(y_vals, y_space)
                    all_vals[r2] = [a, b, c]
        keys = all_vals.keys()
        maximum_r2 = max(keys)
        self.init_constants[0] = all_vals[maximum_r2][0]
        self.init_constants[1] = all_vals[maximum_r2][1]
        self.init_constants[2] = all_vals[maximum_r2][2]

    # IN-USE
    def implement_curve_fit(self, x_vals, y_vals, re_eval_inits: bool):
        if re_eval_inits or self.re_eval_init_constants:
            self.set_init_constants(x_vals, y_vals)
        opt_constants, _ = curve_fit(self.buckingham_potential, x_vals, y_vals,
                                     maxfev=10000, p0=self.init_constants,
                                     bounds=((self.a_lower_bound_cf, self.b_lower_bound_cf,
                                              self.c_lower_bound_cf),
                                             (self.a_upper_bound_cf, self.b_upper_bound_cf,
                                              self.c_upper_bound_cf)))
        return opt_constants

    # IN-USE
    def evaluate_parameters(self, frame_to_eval: pd.DataFrame, re_eval_inits: bool):
        """
        Evaluate Buckingham potential constants
        :param re_eval_inits:
        :param frame_to_eval: Particular Element - PC combination being evaluated
        :return:
        """
        x_vals = frame_to_eval['Distance (A)']
        y_vals = frame_to_eval['Relative LJ Enrgy']
        de_constants = self.implement_diff_evolution(x_vals, y_vals)
        cf_constants = self.implement_curve_fit(x_vals, y_vals, re_eval_inits)

        a_de, b_de, c_de = de_constants[0], de_constants[1], de_constants[2]
        a_cv, b_cv, c_cv = cf_constants

        y_space_de = self.buckingham_potential(x_vals, a_de, b_de, c_de)
        y_space_cv = self.buckingham_potential(x_vals, a_cv, b_cv, c_cv)

        r2_de = r2_score(y_vals, y_space_de)
        r2_cv = r2_score(y_vals, y_space_cv)

        if r2_de >= r2_cv:
            return r2_de, a_de, b_de, c_de
        else:
            return r2_cv, a_cv, b_cv, c_cv

    def set_basic_bounds(self):
        self.a_bounds_de = [self.a_lower, self.a_upper]
        self.b_bounds_de = [self.b_lower, self.b_upper]
        self.c_bounds_de = [self.c_lower, self.c_upper]
        self.a_lower_bound_cf = self.a_lower
        self.a_upper_bound_cf = self.a_upper
        self.b_lower_bound_cf = self.b_lower
        self.b_upper_bound_cf = self.b_upper
        self.c_lower_bound_cf = self.c_lower
        self.c_upper_bound_cf = self.c_upper

    def set_firm_bounds(self, a, b, c):
        self.a_bounds_de = [a - 20, a + 20]
        # self.b_bounds_de = [b - 0.002, b + 0.002]
        self.b_bounds_de = [0, b + 0.002]
        self.c_bounds_de = [c - 0.001, c + 0.001]
        self.a_lower_bound_cf = a - 20
        self.a_upper_bound_cf = a + 20
        # self.b_lower_bound_cf = b - 0.002
        self.b_lower_bound_cf = 0
        self.b_upper_bound_cf = b + 0.002
        self.c_lower_bound_cf = c - 0.0005
        self.c_upper_bound_cf = c + 0.0005

    def set_eval_data(self, frame: pd.DataFrame, r2, a, b, c, num_records, main_data: bool):
        """
        Add the evaluated R2 values and Buckingham potential constants to the frame
        :param main_data:
        :param num_records:
        :param frame: frame to append to
        :param r2: R2 value
        :param a: A value
        :param b: B value
        :param c: C value
        :return: None
        """
        data_row = [frame['Element Name'].iloc[0],
                    frame['Partial Charge'].iloc[0],
                    frame['Atomic Mass'].iloc[0],
                    frame['Atomic Radius'].iloc[0],
                    frame['Atomic Number'].iloc[0],
                    r2, a, b, c, num_records]
        if main_data:
            self.master_results_frame.loc[len(self.master_results_frame)] = data_row
        else:
            self.temp_results_frame.loc[len(self.temp_results_frame)] = data_row

    def re_eval_on_max_r2(self):
        # Find A, B, C for the best R2
        max_r2_index = self.temp_results_frame[['R2']].idxmax()
        a = self.temp_results_frame.loc[max_r2_index, 'A'].values[0]
        b = self.temp_results_frame.loc[max_r2_index, 'B'].values[0]
        c = self.temp_results_frame.loc[max_r2_index, 'C'].values[0]

        # Set bounds around that A, B, C
        self.set_firm_bounds(a, b, c)

        # Get all frames for this element now
        this_element_frames = self.get_element_pc_frame(self.element_state)
        self.re_eval_init_constants = True
        for frame in this_element_frames.copy():
            num_rec = len(frame)
            r2, a, b, c = self.evaluate_parameters(frame, False)
            self.set_eval_data(frame, r2, a, b, c, num_rec, True)
            self.re_eval_init_constants = False
        self.set_basic_bounds()
        print('BREAK')

    def set_r_squared_data(self, element=None, partial_charge=None):
        """

        :param element:
        :param partial_charge:
        :return:
        """
        # We have the raw data from simulations already in one sheet named 'raw_data.xlsx'
        # We retrieve each Element-PC combination as a dataframe and save that dataframe as an item in array
        usable_frame = self.get_element_pc_frame(element, partial_charge)
        print('STOPPAGE AT', len(usable_frame))
        if element is None and partial_charge is None:
            # Start first Element-PC combo
            count = 0
            for given_frame in usable_frame.copy():
                current_element = given_frame['Element Name'].iloc[0]
                print('Element is ' + given_frame['Element Name'].iloc[0])
                print('Partial Charge is ' + str(given_frame['Partial Charge'].iloc[0]))
                num_rec = len(given_frame)
                if (self.element_state is not None and self.element_state != current_element) or (
                        self.element_state is None and self.element_state != current_element):
                    if (self.element_state is not None and self.element_state != current_element) or count == len(
                            usable_frame):
                        ################### EVALUATING NEW CONSTANTS KEEPING THE BEST R2 FOR AN ELEMENT ################
                        self.re_eval_on_max_r2()
                        self.temp_results_frame = self.temp_results_frame.iloc[0:0]
                        ####################################### END OF RE-EVALUATION ###################################
                    r2, a, b, c = self.evaluate_parameters(given_frame, True)
                    self.set_eval_data(given_frame, r2, a, b, c, num_rec, False)
                    self.element_state = current_element
                    print('BREAK')
                else:
                    r2, a, b, c = self.evaluate_parameters(given_frame, False)
                    self.set_eval_data(given_frame, r2, a, b, c, num_rec, False)
                    print('BREAK')
                count += 1
                print(count)

    # All the results have been updated on the master frame, convert that frame to an Excel sheet
    def export_master_frame_to_sheet(self):
        self.master_results_frame.to_excel('final_data.xlsx')
