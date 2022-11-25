import os
import pickle
import re
import traceback
from collections import defaultdict
from pathlib import Path, PosixPath
import pandas as pd
from mendeleev import element


class RawDataCreation:
    def __init__(self):
        """
        - 0-Arg constructor for the class
        - Initialises master frame to be exported to excel finally
        - Initialises a dict of all elements and their partial charges to be pickled later
        """
        self.master_frame = pd.DataFrame()
        self.sim_results = list(Path("./sim_data/").rglob("*.[xX][lL][sS][xX]"))
        self.element_pc_dict = defaultdict(list)

    @staticmethod
    def get_element_name(file_path: PosixPath) -> str:
        """
        Gives the name of the element from the sheet
        :param file_path: Path of sheet
        :return: name of the element
        """
        path_list = os.path.split(file_path)
        element_name = path_list[0].split(os.path.sep)[1]
        return element_name

    @staticmethod
    def get_atomic_mass(element_name: str):
        """
        Returns atomic mass of the element from the extracted element name
        :param element_name: Element name
        :return: Atomic mass
        """
        si = element(element_name)
        return si.atomic_weight

    @staticmethod
    def get_atomic_radius(element_name: str):
        """
        Returns atomic radius of the element from the extracted element name
        :param element_name: Element Name
        :return: Atomic Radius
        """
        si = element(element_name)
        ar = si.covalent_radius
        return ar

    @staticmethod
    def get_atomic_number(element_name: str):
        si = element(element_name)
        an = si.atomic_number
        return an

    @staticmethod
    def get_ion_energy(element_name: str):
        """
        Returns first ionization energy of the element from the extracted element name
        :param element_name: Element Name
        :return: Ionization Energy
        """
        si = element(element_name)
        return si.ionization_energies[0]

    @staticmethod
    def get_partial_charge(file_path: PosixPath) -> str:
        """
        Returns the partial charge being worked on from the file name
        :param file_path: Path of the file
        :return: Partial charge
        """
        init_pc = os.path.split(file_path)[1]
        exact_pc = re.search('([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?', init_pc)
        return exact_pc.group(1)

    def create_master_excel(self):
        """
        - Pulls relevant data from each simulation file and stores it in a data frame
        - Also adds relevant required info for any particular simulation sheet's element
        - Adds element and corresponding partial charge for referring to later
        - Exports the full data frame generated into an Excel sheet
        :return: None
        """

        for file in self.sim_results:
            try:
                ele_data_frame = pd.read_excel(file, sheet_name=0)

                element_name = self.get_element_name(file_path=file)
                ele_data_frame['Element Name'] = element_name

                partial_charge = self.get_partial_charge(file_path=file)
                ele_data_frame['Partial Charge'] = partial_charge

                atomic_mass = self.get_atomic_mass(element_name)
                ele_data_frame['Atomic Mass'] = atomic_mass

                atomic_radius = self.get_atomic_radius(element_name)
                ele_data_frame['Atomic Radius'] = atomic_radius

                atomic_number = self.get_atomic_number(element_name)
                ele_data_frame['Atomic Number'] = atomic_number

                self.element_pc_dict[element_name].append(partial_charge)
                self.master_frame = pd.concat([self.master_frame, ele_data_frame], ignore_index=True)
            except BaseException:
                traceback.print_exc()

        with open('element_pc_dict.pkl', 'wb') as f:
            pickle.dump(self.element_pc_dict, f)
        self.master_frame.to_excel('raw_data.xlsx')
