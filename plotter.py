import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from perceptron import Perceptron
from constants import *


class Plotter:
    X, Y = np.array([]), []
    perceptron = None
    learning_rate = 0
    max_epochs = 0
    current_epoch = 0
    current_epoch_text = None
    weights_initialized = False
    perceptron_fitted = False
    decision_boundary = None
    done = False

    def __init__(self):
        self.fig, (self.ax_main, self.ax_errors) = plt.subplots(1, 2)
        self.fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT, forward=True)
        plt.subplots_adjust(bottom=0.3)
        self.ax_main.set_xlim(NORMALIZATION_RANGE)
        self.ax_main.set_ylim(NORMALIZATION_RANGE)
        self.fig.suptitle(FIG_SUPERIOR_TITLE)
        self.ax_main.set_title(MAIN_SUBPLOT_TITLE)
        self.ax_errors.set_title(ERRORS_SUBPLOT_TITLE)
        self.ax_errors.set_xlabel(ERRORS_SUBPLOT_XLABEL)
        self.ax_errors.set_ylabel(ERRORS_SUBPLOT_YLABEL)

        ax_text_box_learning_rate = plt.axes(TEXT_BOX_LEARNING_RATE_AXES)
        ax_text_box_max_epochs = plt.axes(TEXT_BOX_MAX_EPOCHS_AXES)
        ax_button_weights = plt.axes(BUTTON_WEIGHTS_AXES)
        ax_button_perceptron = plt.axes(BUTTON_PERCEPTRON_AXES)
        self.text_box_learning_rate = TextBox(ax_text_box_learning_rate, TEXT_BOX_LEARNING_RATE_PROMPT)
        self.text_box_max_epochs = TextBox(ax_text_box_max_epochs, TEXT_BOX_MAX_EPOCHS_PROMPT)
        button_weights = Button(ax_button_weights, BUTTON_WEIGHTS_TEXT)
        button_perceptron = Button(ax_button_perceptron, BUTTON_PERCEPTRON_TEXT)
        self.text_box_max_epochs.on_submit(self.__submit_max_epochs)
        self.text_box_learning_rate.on_submit(self.__submit_learning_rate)
        button_weights.on_clicked(self.__initialize_weights)
        button_perceptron.on_clicked(self.__fit_perceptron)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()

    def __initialize_weights(self, event):
        learning_rate_initialized = self.learning_rate != 0
        max_epochs_initialized = self.max_epochs != 0
        points_plotted = len(self.X) > 0
        if learning_rate_initialized and max_epochs_initialized and points_plotted and not self.perceptron_fitted:
            self.perceptron = Perceptron(self.learning_rate, self.max_epochs, NORMALIZATION_RANGE)
            self.perceptron.init_weights()
            self.weights_initialized = True
            self.plot_decision_boundary()

    def __fit_perceptron(self, event):
        if self.weights_initialized and not self.perceptron_fitted:
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
                        self.plot_decision_boundary()
                self.__plot_errors(errors)
            self.ax_main.text(PERCEPTRON_CONVERGENCE_TEXT_X_POS, PERCEPTRON_CONVERGENCE_TEXT_Y_POS,
                              PERCEPTRON_CONVERGED_TEXT if self.done else PERCEPTRON_DIDNT_CONVERGE_TEXT,
                              fontsize=PERCEPTRON_CONVERGENCE_TEXT_FONT_SIZE)
            self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
            plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)
            self.perceptron_fitted = True

    def plot_decision_boundary(self):
        x1 = np.array([self.X[:, 0].min() - 2, self.X[:, 0].max() + 2])
        m = -self.perceptron.weights[1] / self.perceptron.weights[2]
        c = self.perceptron.weights[0] / self.perceptron.weights[2]
        x2 = m * x1 + c
        # Plotting
        if not self.decision_boundary:
            self.decision_boundary, = self.ax_main.plot(x1, x2, DECISION_BOUNDARY_MARKER)
            self.current_epoch_text = self.ax_main.text(CURRENT_EPOCH_TEXT_X_POS, CURRENT_EPOCH_TEXT_Y_POS,
                                                        CURRENT_EPOCH_TEXT % self.current_epoch,
                                                        fontsize=CURRENT_EPOCH_TEXT_FONT_SIZE)
        else:
            self.decision_boundary.set_xdata(x1)
            self.decision_boundary.set_ydata(x2)
            self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
        self.fig.canvas.draw()
        plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)

    def __plot_errors(self, count):
        self.ax_errors.bar(self.current_epoch, count)
        plt.pause(ERRORS_SUBPLOT_PAUSE_INTERVAL)

    def __onclick(self, event):
        if event.inaxes == self.ax_main:
            current_point = [event.xdata, event.ydata]
            if self.perceptron_fitted:
                current_point = [-1] + current_point
                self.ax_main.plot(event.xdata, event.ydata,
                                  CLASS_1_MARKER_POST_FIT if self.perceptron.pw(current_point)
                                  else CLASS_0_MARKER_POST_FIT)
            else:
                self.X = np.append(self.X, current_point).reshape([len(self.X) + 1, 2])
                is_left_click = event.button == 1
                #  Left click = Class 0 - Right click = Class 1
                self.Y.append(0 if is_left_click else 1)
                self.ax_main.plot(event.xdata, event.ydata, CLASS_0_MARKER if is_left_click else CLASS_1_MARKER)
            self.fig.canvas.draw()

    def __check_if_valid_expression(self, expression, default_value):
        value = 0
        try:
            value = eval(expression)
        except (SyntaxError, NameError):
            if expression:
                value = default_value
                text_box = self.text_box_learning_rate if default_value == LEARNING_RATE else self.text_box_max_epochs
                text_box.set_val(value)
        finally:
            return value

    def __submit_learning_rate(self, expression):
        self.learning_rate = self.__check_if_valid_expression(expression, LEARNING_RATE)

    def __submit_max_epochs(self, expression):
        self.max_epochs = self.__check_if_valid_expression(expression, MAX_EPOCHS)


if __name__ == '__main__':
    Plotter()
