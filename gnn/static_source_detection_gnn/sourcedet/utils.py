# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt

def one_hot(size, index):
    # Create a list of zeros with the given size
    one_hot_list = [0] * size
    # Set the value at the specified index to 1
    one_hot_list[index] = 1
    return one_hot_list

def log_sum_exp(distr):
	# The predictions (distr) are on log-basis. Therefore, we apply
	# the log-sum-exp trick here. For this we first get the max. value of the log-lik.
	B = np.nanmax(distr)
	# This then computes the normalized probability distribution.
	distr_normalized = torch.exp(distr - (B + torch.log(torch.sum(torch.exp(distr - B)))))
	# Return the normalized distribution.
	return distr_normalized

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_train_valid_curves(train_list, valid_list, dirname):
	# For x-axis
	epochs = range(1, len(train_list) + 1)

	# Plotting of training and validation curves.
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

	# First subplot: Losses
	ax1.plot(epochs, [l[1] for l in train_list], label='Training Loss', marker='o', color='blue')
	ax1.plot(epochs, [l[1] for l in valid_list], label='Validation Loss', marker='o', color='orange')
	ax1.set_title('Loss over Epochs')
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss')
	ax1.legend()
	ax1.grid(True)

	# Second subplot: Accuracies
	ax2.plot(epochs, [l[0] for l in train_list], label='Training Accuracy', marker='o', color='blue')
	ax2.plot(epochs, [l[0] for l in valid_list], label='Validation Accuracy', marker='o', color='orange')
	ax2.set_title('Accuracy over Epochs')
	ax2.set_xlabel('Epochs')
	ax2.set_ylabel('Accuracy')
	ax2.legend()
	ax2.grid(True)

	# Adjust layout so the subplots don't overlap
	plt.tight_layout()

	# Save the plot to a file (e.g., loss_curve.png)
	plt.savefig(dirname + '/loss_acc_curves.png', dpi=300)

	# Close the plot to avoid displaying it
	plt.close()
