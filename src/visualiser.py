import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt
import numpy as np
from src.data_processor import DigitDataProcessor

class DataVisualiser:
    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = DigitDataProcessor.load_data()

    def plot_random_samples(self, num_rows=4, num_cols=4):
        fig, axs = plt.subplots(num_rows, num_cols)
        for ax in axs.flatten():
            idx = np.random.randint(0, len(self.X_train))
            ax.imshow(self.X_train[idx], cmap='gray')
            ax.set_title(self.y_train[idx])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout(pad=0)
        plt.show()

    def plot_digit_examples(self):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()
        idx = np.random.randint(0, len(self.X_train), size=10)
        for i in range(10):
            axes[i].imshow(self.X_train[idx[i]].reshape(28, 28), cmap='Greys')
            axes[i].axis('off')
            axes[i].set_title(str(int(self.y_train[idx[i]])), color='black', fontsize=25)
        plt.show()

    def plot_specific_image(self, index):
        plt.imshow(self.X_train[index].reshape(28, 28), cmap="Greys", interpolation="None")
        plt.title(f"Label: {self.y_train[index]}")
        plt.xticks([])
        plt.yticks([])
        plt.show()


