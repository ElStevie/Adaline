import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import TextBox, Button

from perceptron import Perceptron
from adaline import Adaline
from constants import *


class Plotter:
    X, Y = np.array([]), []
    perceptron = None
    adaline = None
    learning_rate = 0
    max_epochs = 0
    current_epoch = 0
    current_epoch_text = None
    algorithm_convergence_text = None
    perceptron_weights_initialized = False
    adaline_weights_initialized = False
    perceptron_fitted = False
    adaline_fitted = False
    perceptron_decision_boundary = None
    adaline_decision_boundary = None
    perceptron_errors = None
    adaline_errors = None
    done = False

    def __init__(self):
        self.fig, (self.ax_main, self.ax_perceptron_errors, self.ax_adaline_errors) = plt.subplots(SUBPLOT_ROWS,
                                                                                                   SUBPLOT_COLS)
        self.fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT, forward=True)
        plt.subplots_adjust(bottom=0.3)
        self.ax_main.set_xlim(NORMALIZATION_RANGE)
        self.ax_main.set_ylim(NORMALIZATION_RANGE)
        self.fig.suptitle(FIG_SUPERIOR_TITLE)
        self.ax_main.set_title(MAIN_SUBPLOT_PERCEPTRON_TITLE)
        self.ax_perceptron_errors.set_title(PERCEPTRON_ERRORS_SUBPLOT_TITLE)
        self.ax_perceptron_errors.set_xlabel(ERRORS_SUBPLOT_XLABEL)
        self.ax_perceptron_errors.set_ylabel(PERCEPTRON_ERRORS_SUBPLOT_YLABEL)
        self.ax_adaline_errors.set_title(ADALINE_ERRORS_SUBPLOT_TITLE)
        self.ax_adaline_errors.set_xlabel(ERRORS_SUBPLOT_XLABEL)
        self.ax_adaline_errors.set_ylabel(ADALINE_ERRORS_SUBPLOT_YLABEL)

        ax_text_box_learning_rate = plt.axes(TEXT_BOX_LEARNING_RATE_AXES)
        ax_text_box_max_epochs = plt.axes(TEXT_BOX_MAX_EPOCHS_AXES)
        ax_text_box_desired_error = plt.axes(TEXT_BOX_DESIRED_ERROR_AXES)
        ax_button_weights = plt.axes(BUTTON_WEIGHTS_AXES)
        ax_button_perceptron = plt.axes(BUTTON_PERCEPTRON_AXES)
        ax_button_adaline = plt.axes(BUTTON_ADALINE_AXES)
        self.text_box_learning_rate = TextBox(ax_text_box_learning_rate, TEXT_BOX_LEARNING_RATE_PROMPT)
        self.text_box_max_epochs = TextBox(ax_text_box_max_epochs, TEXT_BOX_MAX_EPOCHS_PROMPT)
        self.text_box_desired_error = TextBox(ax_text_box_desired_error, TEXT_BOX_DESIRED_ERROR_PROMPT)
        button_weights = Button(ax_button_weights, BUTTON_WEIGHTS_TEXT)
        button_perceptron = Button(ax_button_perceptron, BUTTON_PERCEPTRON_TEXT)
        button_adaline = Button(ax_button_adaline, BUTTON_ADALINE_TEXT)
        self.text_box_max_epochs.on_submit(self.__submit_max_epochs)
        self.text_box_learning_rate.on_submit(self.__submit_learning_rate)
        self.text_box_desired_error.on_submit(self.__submit_desired_error)
        button_weights.on_clicked(self.__initialize_weights)
        button_perceptron.on_clicked(self.__fit_perceptron)
        button_adaline.on_clicked(self.__fit_adaline)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()

    def __initialize_weights(self, event):
        learning_rate_initialized = self.learning_rate != 0
        max_epochs_initialized = self.max_epochs != 0
        points_plotted = len(self.X) > 0
        if learning_rate_initialized and max_epochs_initialized and points_plotted:
            if self.perceptron_fitted:
                self.adaline = Adaline(self.learning_rate, self.max_epochs, NORMALIZATION_RANGE)
                self.adaline.init_weights()
                self.adaline_weights_initialized = True
                self.plot_decision_boundary(self.adaline)
            else:
                self.perceptron = Perceptron(self.learning_rate, self.max_epochs, NORMALIZATION_RANGE)
                self.perceptron.init_weights()
                self.perceptron_weights_initialized = True
                self.plot_decision_boundary(self.perceptron)

    def __fit_perceptron(self, event):
        if self.perceptron_weights_initialized and not self.perceptron_fitted:
            while not self.done and self.current_epoch < self.perceptron.max_epochs:
                self.done = True
                self.current_epoch += 1
                errors = 0
                for i, x in enumerate(self.X):
                    x = np.insert(x, 0, -1.0)
                    error = self.Y[i] - self.perceptron.pw(x)
                    if error != 0:
                        errors += 1
                        self.done = False
                        self.perceptron.weights = \
                            self.perceptron.weights + np.multiply((self.perceptron.learning_rate * error), x)
                        self.plot_decision_boundary(self.perceptron)
                self.__plot_perceptron_errors(errors)
            self.algorithm_convergence_text = self.ax_main.text(PERCEPTRON_CONVERGENCE_TEXT_X_POS,
                                                                PERCEPTRON_CONVERGENCE_TEXT_Y_POS,
                                                                ALGORITHM_CONVERGED_TEXT if self.done else
                                                                ALGORITHM_DIDNT_CONVERGE_TEXT,
                                                                fontsize=PERCEPTRON_CONVERGENCE_TEXT_FONT_SIZE)
            self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
            plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)
            self.perceptron_fitted = True
            self.current_epoch = 0
            self.ax_main.set_title(MAIN_SUBPLOT_ADALINE_TITLE)
            self.algorithm_convergence_text.set_text(None)

    def __fit_adaline(self, event):
        if not self.adaline_fitted and self.adaline_weights_initialized and self.desired_error != 0.0:
            cumulative_error = 1
            while cumulative_error > self.desired_error and self.current_epoch < self.max_epochs:
                self.current_epoch += 1
                cumulative_error = 0
                for i, x in enumerate(self.X):
                    x = np.insert(x, 0, -1.0)
                    learning_rate = self.learning_rate * 2
                    f_y = self.adaline.fw(x)
                    der_f_y = f_y * (1.0 - f_y)
                    error = self.Y[i] - f_y
                    cumulative_error += error ** 2
                    self.adaline.weights = \
                        self.adaline.weights + np.multiply((learning_rate * error * der_f_y), x)
                    self.plot_decision_boundary(self.adaline)
                self.__plot_adaline_errors(cumulative_error)
            self.plot_decision_regions(self.adaline)
            plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)
            self.adaline_fitted = True

    def plot_decision_regions(self, perceptron, resolution=0.02):
        markers = ('x', '.', '^', 'v')
        colors = ('b', 'r', 'g', 'k', 'grey')
        cmap = ListedColormap(colors[:len(np.unique(self.Y))])

        # plot the decision regions by creating a pair of grid arrays xx1 and xx2 via meshgrid function in Numpy
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

        # predict the class labels z of the grid points
        Z = np.array([perceptron.pw(np.insert(x, 0, -1)) for x in np.array([xx1.ravel(), xx2.ravel()]).T])
        Z = Z.reshape(xx1.shape)

        # draw the contour using matplotlib
        self.ax_main.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

        # plot class samples
        for i, cl in enumerate(np.unique(self.Y)):
            self.ax_main.scatter(x=self.X[self.Y == cl, 0], y=self.X[self.Y == cl, 1],
                                 alpha=0.8, c=cmap(i), marker=markers[i], label=cl)

    def plot_decision_boundary(self, perceptron):
        x1 = np.array([self.X[:, 0].min() - 2, self.X[:, 0].max() + 2])
        m = -perceptron.weights[1] / perceptron.weights[2]
        c = perceptron.weights[0] / perceptron.weights[2]
        x2 = m * x1 + c
        # Plotting
        is_perceptron = type(perceptron) == Perceptron
        if is_perceptron:
            if self.perceptron_decision_boundary:
                self.perceptron_decision_boundary.set_xdata(x1)
                self.perceptron_decision_boundary.set_ydata(x2)
                self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
            else:
                self.perceptron_decision_boundary, = self.ax_main.plot(x1, x2, PERCEPTRON_DECISION_BOUNDARY_MARKER)
                self.current_epoch_text = self.ax_main.text(CURRENT_EPOCH_TEXT_X_POS, CURRENT_EPOCH_TEXT_Y_POS,
                                                            CURRENT_EPOCH_TEXT % self.current_epoch,
                                                            fontsize=CURRENT_EPOCH_TEXT_FONT_SIZE)
        else:
            if self.adaline_decision_boundary:
                self.adaline_decision_boundary.set_xdata(x1)
                self.adaline_decision_boundary.set_ydata(x2)
            else:
                self.adaline_decision_boundary, = self.ax_main.plot(x1, x2, ADALINE_DECISION_BOUNDARY_MARKER)
            self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
        self.fig.canvas.draw()
        plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL if is_perceptron else MAIN_SUBPLOT_ADALINE_PAUSE_INTERVAL)

    def __plot_perceptron_errors(self, count):
        if not self.perceptron_errors:
            self.perceptron_errors = [[], []]
        else:
            self.ax_perceptron_errors.clear()
        self.perceptron_errors[0].append(self.current_epoch)
        self.perceptron_errors[1].append(count)
        self.ax_perceptron_errors.plot(self.perceptron_errors[0], self.perceptron_errors[1],
                                       PERCEPTRON_DECISION_BOUNDARY_MARKER)
        plt.pause(ERRORS_SUBPLOT_PAUSE_INTERVAL)

    def __plot_adaline_errors(self, cumulative_error):
        if not self.adaline_errors:
            self.adaline_errors = [[], []]
        else:
            self.ax_adaline_errors.clear()
        self.adaline_errors[0].append(self.current_epoch)
        self.adaline_errors[1].append(cumulative_error)
        self.ax_adaline_errors.plot(self.adaline_errors[0], self.adaline_errors[1], ADALINE_DECISION_BOUNDARY_MARKER)
        plt.pause(ERRORS_SUBPLOT_PAUSE_INTERVAL)

    def __onclick(self, event):
        if event.inaxes == self.ax_main:
            current_point = [event.xdata, event.ydata]
            is_left_click = event.button == 1
            if self.perceptron_fitted:
                if is_left_click:
                    current_point = [-1] + current_point
                    self.ax_main.plot(event.xdata, event.ydata,
                                      CLASS_1_MARKER_POST_PERCEPTRON_FIT if self.perceptron.pw(current_point)
                                      else CLASS_0_MARKER_POST_PERCEPTRON_FIT)
                else:
                    if self.adaline_fitted:
                        current_point = [-1] + current_point
                        self.ax_main.plot(event.xdata, event.ydata,
                                          CLASS_1_MARKER_POST_ADALINE_FIT if self.adaline.pw(current_point)
                                          else CLASS_0_MARKER_POST_ADALINE_FIT)
            else:
                self.X = np.append(self.X, current_point).reshape([len(self.X) + 1, 2])
                #  Left click = Class 0 - Right click = Class 1
                self.Y.append(0 if is_left_click else 1)
                self.ax_main.plot(event.xdata, event.ydata, CLASS_0_MARKER if is_left_click else CLASS_1_MARKER)
            self.fig.canvas.draw()

    def __check_if_valid_expression(self, expression, text_box, default_value):
        value = 0
        try:
            value = eval(expression)
        except (SyntaxError, NameError):
            if expression:
                value = default_value
                text_box.set_val(value)
        finally:
            return value

    def __submit_learning_rate(self, expression):
        self.learning_rate = self.__check_if_valid_expression(expression, self.text_box_learning_rate, LEARNING_RATE)

    def __submit_max_epochs(self, expression):
        self.max_epochs = self.__check_if_valid_expression(expression, self.text_box_max_epochs, MAX_EPOCHS)

    def __submit_desired_error(self, expression):
        self.desired_error = self.__check_if_valid_expression(expression, self.text_box_desired_error, DESIRED_ERROR)


if __name__ == '__main__':
    Plotter()
