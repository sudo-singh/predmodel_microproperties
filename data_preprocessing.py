import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreProcessing:
    def __init__(self):
        """
        - 0-Arg constructor for this class
        - Sets the main frame of data to be worked on
        """
        self.main_frame = pd.read_excel('final_data.xlsx', sheet_name=0)
        self.unseen_input_frame = pd.read_excel('MD_Simulation_Data.xlsx', sheet_name=0)
        self.working_input_frame = None
        self.working_frame: pd.DataFrame = None
        self.x_block = None
        self.y_block = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None

    def set_working_frame(self):
        """
        - Drops irrelevant columns from the main working frame
        - R2 values and element names from Initial data are removed
        :return: None
        """
        self.working_frame = self.main_frame.copy()
        self.working_frame.drop('Element Name', axis=1, inplace=True)
        self.working_frame.drop('R2', axis=1, inplace=True)
        self.working_frame.drop('Records', axis=1, inplace=True)
        self.working_frame = self.working_frame.loc[:, ~self.working_frame.columns.str.contains('^Unnamed')]

    def set_unseen_frame(self):
        """
        - Drops irrelevant columns from the main working frame
        - R2 values and element names from Initial data are removed
        :return: None
        """
        self.working_input_frame = self.unseen_input_frame.copy()
        self.working_input_frame.drop('Element Name', axis=1, inplace=True)
        self.working_input_frame = self.working_input_frame.loc[:, ~self.working_input_frame.columns.str.contains('^Unnamed')]
        print('Break')

    def feature_label_splitter(self):
        """
        - Splits the Feature block and Label block from the data
        :return: None
        """
        self.x_block = self.working_frame.copy()
        self.x_block.drop(['A', 'B', 'C'], axis=1, inplace=True)
        self.y_block = self.working_frame.copy()
        self.y_block.drop(['Atomic Mass', 'Partial Charge', 'Atomic Radius', 'Atomic Number'],
                          axis=1, inplace=True)

    def test_train_split(self):
        """
        - Splits training-testing data using sklearn's library
        :return: None
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_block, self.y_block,
                                                                                random_state=32, test_size=0.25,
                                                                                shuffle=True)

    def standard_normalize(self):
        """
        - Mean centers train data
        - Unit scale train data
        - use the transformation scale on the test data
        :return: None
        """
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        print('PRINT ALL')
        print(self.scaler.mean_)
        print(self.scaler.scale_)

    def normalize_unseen(self):
        self.working_input_frame = self.scaler.transform(self.working_input_frame)
        print('Break')

    def export_working_data_to_excel(self):
        """
        - Converts the final working frame to sheet if needed for checking data
        :return: None
        """
        self.working_frame.to_excel('working_data.xlsx')

    def export_results(self):
        self.unseen_input_frame.to_excel('results.xlsx')
