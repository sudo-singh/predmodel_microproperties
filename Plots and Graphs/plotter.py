import matplotlib
import matplotlib.pyplot as plt
import numpy.random
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, patches
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import randrange


class Plotter:
    def __init__(self):
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 12
        self.markers = ['d', 'X', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p']
        self.marker_size = 10
        self.width, self.height = plt.figaspect(1)
        self.fig = plt.figure(figsize=(self.width, self.height), dpi=1000, tight_layout=False)

    def element_data(self):
        main_frame = pd.read_excel('O. -3.0.xlsx', sheet_name=0)
        main_frame = main_frame.sort_values("Distance (A)")
        plt.plot(main_frame['Distance (A)'], main_frame['Relative LJ Enrgy'], label='Oxygen -3.0 Data', c='blue',
                 linestyle='dashed', linewidth=1)
        plt.scatter(main_frame['Distance (A)'], main_frame['Relative LJ Enrgy'], s=self.marker_size, c='red')

        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=12)
        plt.legend(loc='upper right', prop=font)
        plt.xlabel('Distance')
        plt.ylabel('Energy')
        plt.title("Energy v/s Distance [Oxygen -3.0]", fontweight='bold')
        plt.savefig('../Paper Images/Oxygen_Bad.png')
        plt.show()

    def clustered_constants(self):
        main_frame = pd.read_excel('final_data.xlsx', sheet_name=0)
        fig = plt.figure(figsize=(15, 10), dpi=1000)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        # cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=12)
        final_data = main_frame.loc[
            main_frame['Element Name'].isin(['Hydrogen', 'Carbon', 'Magnesium', 'Phosphorus', 'Potassium',
                                             'Arsenic', 'Strontium', 'Palladium', 'Antimony', 'Technetium'
                                                , 'Ruthenium'])]
        # elements = np.unique(final_data['Element Name'])
        # colors = np.linspace(0, 1, len(elements))
        # colordict = dict(zip(elements, colors))
        # final_data["Color"] = final_data['Element Name'].apply(lambda x: colordict[x])
        # print('Break')
        # sc = ax.scatter(final_data['A'], final_data['B'], final_data['C'], s=self.marker_size, marker='o',
        #                 cmap=cmap, c=final_data.Color, alpha=1)

        groups = final_data.groupby('Element Name')
        elements_shown = list(groups.groups.keys())
        handles = []
        count = 0
        for name, group in groups:
            line, = ax.plot(group.A, group.B, group.C, marker=self.markers[count],
                            label=name, linestyle='None', markersize=15)
            count += 1
            handles.append(line)
        ax.set_xlabel('A', fontsize=16, labelpad=19)
        ax.set_ylabel('B', fontsize=16, labelpad=19)
        ax.set_zlabel('C', fontsize=16, labelpad=19)
        ax.tick_params(which='major', labelsize=15, pad=11)
        first_legend = ax.legend(handles=[handles[0], handles[1], handles[2], handles[6], handles[7], handles[8]], loc=1, bbox_to_anchor=(1.0, 1.0), prop=font)
        second_legend = ax.legend(handles=[handles[3], handles[4], handles[5], handles[9], handles[10]], loc=2, bbox_to_anchor=(0.0, 1.0), prop=font)
        # third_legend = ax.legend(handles=[handles[6], handles[7], handles[8]], loc=3, bbox_to_anchor=(0.50, 0.95), prop=font)
        # fourth_legend = ax.legend(handles=[handles[9], handles[10]], loc=4, bbox_to_anchor=(0.45, 0.95), prop=font)
        plt.gca().add_artist(first_legend)
        plt.gca().add_artist(second_legend)
        # plt.gca().add_artist(third_legend)
        # plt.gca().add_artist(fourth_legend)
        # plt.suptitle('Clustered Constants after setting bounds', fontweight='bold', fontsize=18)
        plt.savefig('../Paper Images/Clustered_Constants.png', bbox_inches='tight')
        plt.show()
        print('Break')

    def non_clustered_constants(self):
        main_frame = pd.read_excel('non_clustered_data.xlsx', sheet_name=0)
        fig = plt.figure(figsize=(15, 10), dpi=1000)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=12)
        final_data = main_frame.loc[
            main_frame['Element Name'].isin(['Hydrogen', 'Carbon', 'Magnesium', 'Phosphorus', 'Potassium',
                                             'Arsenic', 'Strontium', 'Palladium', 'Antimony', 'Technetium',
                                             'Ruthenium'])]
        groups = final_data.groupby('Element Name')
        elements_shown = list(groups.groups.keys())
        handles = []
        count = 0
        for name, group in groups:
            line, = ax.plot(group.A, group.B, group.C, label=name, marker=self.markers[count], linestyle='None',
                            markersize=15)
            count += 1
            # ax.scatter(group.A, group.B, group.C, s=5, label=name)
            handles.append(line)
        ax.set_xlabel('A', fontsize=16, labelpad=19)
        ax.set_ylabel('B', fontsize=16, labelpad=19)
        ax.set_zlabel('C', fontsize=16, labelpad=19)
        ax.tick_params(which='major', labelsize=15, pad=11)
        first_legend = ax.legend(handles=[handles[0], handles[1], handles[2], handles[6], handles[7], handles[8]], loc='upper left', bbox_to_anchor=(1.0, 1.0), prop=font)
        second_legend = ax.legend(handles=[handles[3], handles[4], handles[5], handles[9], handles[10]], loc='upper right', bbox_to_anchor=(0.0, 1.0), prop=font)
        # third_legend = ax.legend(handles=[handles[6], handles[7], handles[8]], loc='lower left', bbox_to_anchor=(0.50, 0.95), prop=font)
        # fourth_legend = ax.legend(handles=[handles[9], handles[10]], loc='lower right', bbox_to_anchor=(0.45, 0.95), prop=font)
        plt.gca().add_artist(first_legend)
        plt.gca().add_artist(second_legend)
        # plt.gca().add_artist(third_legend)
        # plt.gca().add_artist(fourth_legend)
        # plt.suptitle('Non clustered constants without setting bounds', fontweight='bold', fontsize=18)
        plt.savefig('../Paper Images/Non_Clustered_Constants.png', bbox_inches='tight')
        print('Break')

    @staticmethod
    def compare_final_data():
        main_frame = pd.read_excel('../Result_comparison_two.xlsx', sheet_name=0)
        fig = plt.figure(figsize=(15, 10), dpi=1000)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        groups = main_frame.groupby('Element Name')
        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=12)
        handles = []
        for name, group in groups:
            number = numpy.random.rand(3, )
            line, = ax.plot(group.A, group.B, group.C, label=name,
                            marker='o', linestyle='None', c=number, markersize=10)
            ax.plot(group.A_Actual, group.B_Actual, group.C_Actual,
                    marker='x', linestyle='None', label=name, c=number, markersize=10)
            handles.append(line)
        # plt.suptitle('Actual v/s Expected - Experimental Data', fontweight='bold', fontsize=18)
        ax.set_xlabel('A', fontsize=16, labelpad=19)
        ax.set_ylabel('B', fontsize=16, labelpad=19)
        ax.set_zlabel('C', fontsize=16, labelpad=19)
        ax.tick_params(which='major', labelsize=11)
        legend_elements = [Line2D([0], [0], marker='x', label='ML Technique Based Parameters'),
                           Line2D([0], [0], marker='o', label='Experimental Parameters')]
        first_legend = ax.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4], handles[5]], loc=1, bbox_to_anchor=(1.0, 1.0), prop=font)
        # second_legend = ax.legend(handles=[handles[3], handles[4], handles[5]],
        #                           loc=4, bbox_to_anchor=(0.50, 0.95), prop=font)
        third_legend = ax.legend(handles=legend_elements, loc=2, prop=font, bbox_to_anchor=(0.0, 1.0))
        plt.gca().add_artist(first_legend)
        # plt.gca().add_artist(second_legend)
        plt.gca().add_artist(third_legend)
        plt.savefig('../Paper Images/Actual vs Expected - Final Data.png', bbox_inches='tight')
        plt.show()
        print("Break")

    def element_data_zoom(self):
        main_frame = pd.read_excel('C.+0.0.xlsx', sheet_name=0)
        main_frame2 = pd.read_excel('C.+1.0.xlsx', sheet_name=0)
        main_frame3 = pd.read_excel('C.+2.0.xlsx', sheet_name=0)
        main_frame = main_frame.sort_values("Distance (A)")
        main_frame2 = main_frame2.sort_values("Distance (A)")
        main_frame3 = main_frame3.sort_values("Distance (A)")

        fig, ax = plt.subplots(figsize=(15, 10), dpi=1000)

        ax.plot(main_frame['Distance (A)'], main_frame['Relative LJ Enrgy'], label='Carbon - Carbon +0.0', c='red',
                linestyle='dashed', linewidth=2)
        ax.plot(main_frame2['Distance (A)'], main_frame2['Relative LJ Enrgy'], label='Carbon - Carbon +1.0', c='green',
                linestyle='dashed', linewidth=2)
        ax.plot(main_frame3['Distance (A)'], main_frame3['Relative LJ Enrgy'], label='Carbon - Carbon +2.0', c='blue',
                linestyle='dashed', linewidth=2)
        ax.scatter(main_frame['Distance (A)'], main_frame['Relative LJ Enrgy'], s=self.marker_size, c='red')
        ax.scatter(main_frame2['Distance (A)'], main_frame2['Relative LJ Enrgy'], s=self.marker_size, c='green')
        ax.scatter(main_frame3['Distance (A)'], main_frame3['Relative LJ Enrgy'], s=self.marker_size, c='blue')
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.xlabel('Distance', fontsize=18)
        plt.ylabel('Energy', fontsize=18)
        # plt.title("Energy v/s Distance [Carbon]", fontweight='bold')

        zax = zoomed_inset_axes(ax, 4, loc=10)

        zax.plot(main_frame['Distance (A)'], main_frame['Relative LJ Enrgy'], c='red',
                 linestyle='dashed', linewidth=1.2)
        zax.plot(main_frame2['Distance (A)'], main_frame2['Relative LJ Enrgy'], c='green',
                 linestyle='dashed', linewidth=1.2)
        zax.plot(main_frame3['Distance (A)'], main_frame3['Relative LJ Enrgy'], c='blue',
                 linestyle='dashed', linewidth=1.2)
        zax.scatter(main_frame['Distance (A)'], main_frame['Relative LJ Enrgy'], s=self.marker_size, c='red')
        zax.scatter(main_frame2['Distance (A)'], main_frame2['Relative LJ Enrgy'], s=self.marker_size, c='green')
        zax.scatter(main_frame3['Distance (A)'], main_frame3['Relative LJ Enrgy'], s=self.marker_size, c='blue')

        zax.set_xlim((0.3, 1.5))
        zax.set_xticks((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5))
        zax.set_ylim((-10, 290))
        zax.set_yticks((-10, 40, 90, 140, 190, 240, 290))
        zax.tick_params(axis='both', which='major', labelsize=11)
        zax.set_xlabel('Distance', fontsize=14)
        zax.set_ylabel('Energy', fontsize=14)

        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=14)

        arrow_x_start = (1.5, 0.01)  # Start point of the arrow in ax's coordinates
        arrow_x_end = (6.37, 457)  # End point of the arrow in ax's coordinates
        arrow_x_text = ''  # Text to display next to the arrow

        ax.annotate(arrow_x_text, xy=arrow_x_end, xytext=arrow_x_start,
                    arrowprops=dict(arrowstyle='->', linestyle='--', color='gray', mutation_scale=15, linewidth=0.8))

        # Create the arrow connecting the bottom of zax's y-axis to the main plot ax
        arrow_y_start = (0.3, 290)  # Start point of the arrow in ax's coordinates
        arrow_y_end = (1.54, 1667)  # End point of the arrow in ax's coordinates
        arrow_y_text = ''  # Text to display next to the arrow

        ax.annotate(arrow_y_text, xy=arrow_y_end, xytext=arrow_y_start,
                    arrowprops=dict(arrowstyle='->', linestyle='--', color='gray', mutation_scale=15, linewidth=0.8))

        rect_x_min = 0.3
        rect_x_max = 1.5
        rect_y_min = -10
        rect_y_max = 290

        rect = patches.Rectangle((rect_x_min, rect_y_min), rect_x_max - rect_x_min, rect_y_max - rect_y_min,
                                 linewidth=2, edgecolor='gray', facecolor='none', linestyle='dotted')

        ax.add_patch(rect)

        ax.legend(loc='upper right', prop=font)
        plt.savefig('../Paper Images/Carbon_Good.png', bbox_inches='tight')
        plt.show()
