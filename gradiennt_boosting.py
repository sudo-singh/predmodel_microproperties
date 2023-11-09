from matplotlib import pyplot as plt, font_manager
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from data_preprocessing import DataPreProcessing
from sklearn.svm import SVR


class GradientBoosting(DataPreProcessing):
    def __init__(self):
        plt.style.use('seaborn-whitegrid')
        super().__init__()
        self.test_model = None

    def boost(self):
        param_grid = {
            'estimator__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 160, 190, 200, 210, 220, 250],
            'estimator__learning_rate': [0.04, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]}

        gbr = GradientBoostingRegressor(random_state=0)
        # mor1 = MultiOutputRegressor(gbr)
        self.test_model = GridSearchCV(MultiOutputRegressor(gbr), param_grid=param_grid, verbose=1, cv=7)
        self.test_model.fit(self.x_train, self.y_train)
        gg = self.test_model.score(self.x_train, self.y_train)
        tt = self.test_model.score(self.x_test, self.y_test)
        print("Break")

    def rfr(self):
        param_grid = {
            "estimator__n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 160, 190, 200, 210, 220, 250],
            "estimator__min_samples_split": [2, 4, 6, 8, 10, 12, 14],
            "estimator__bootstrap": [True, False],
        }
        rfr = RandomForestRegressor(random_state=0)
        mor2 = MultiOutputRegressor(rfr)
        self.test_model = GridSearchCV(MultiOutputRegressor(rfr), param_grid=param_grid, verbose=1, cv=7)
        self.test_model.fit(self.x_train, self.y_train)
        ll = self.test_model.score(self.x_test, self.y_test)
        ff = self.test_model.score(self.x_train, self.y_train)
        output_frame_test = self.test_model.predict(self.x_test)
        #
        fig = plt.figure(figsize=(15, 10), dpi=1000)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=12)
        ax.plot(output_frame_test[:, 0], output_frame_test[:, 1], output_frame_test[:, 0], label='Predicted Values',
                marker='o', linestyle='None')
        ax.plot(self.y_test['A'], self.y_test['B'], self.y_test['C'], label='Actual Values',
                marker='d', linestyle='None')
        # plt.suptitle('Actual v/s Expected - Test Data', fontweight='bold', fontsize=18)
        ax.set_xlabel('A', fontsize=16, labelpad=19)
        ax.set_ylabel('B', fontsize=16, labelpad=19)
        ax.set_zlabel('C', fontsize=16, labelpad=19)
        ax.tick_params(which='major', labelsize=11)
        ax.legend(borderaxespad=4, prop=font)
        plt.savefig('./Paper Images/Actual vs Expected - Test Data.png', bbox_inches='tight')
        plt.show()
        print("Break")

    def svr(self):
        parameters = {'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed'],
                      'estimator__C': [1.5, 2.5, 4.5, 7, 9, 10, 12, 13],
                      'estimator__gamma': ['scale', 'auto'],
                      'estimator__epsilon': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]}
        svr = SVR()
        self.test_model = GridSearchCV(MultiOutputRegressor(svr), param_grid=parameters, verbose=1, cv=4)
        self.test_model.fit(self.x_train, self.y_train)
        ll = self.test_model.score(self.x_test, self.y_test)
        ff = self.test_model.score(self.x_train, self.y_train)
        print('Break')

    def predict(self):
        out = self.test_model.predict(self.working_input_frame)
        self.unseen_input_frame['A'] = out[:, 0]
        self.unseen_input_frame['B'] = out[:, 1]
        self.unseen_input_frame['C'] = out[:, 2]
        print('Break')