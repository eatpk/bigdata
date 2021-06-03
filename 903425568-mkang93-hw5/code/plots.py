import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	epochs_losses = np.arange(len(train_losses))
	epochs_acc = np.arange(len(train_accuracies))

	# Plot of Loss Curve
	
	plt.figure()
	plt.plot(epochs_losses, train_losses, label='Training Loss')
	plt.plot(epochs_losses, valid_losses, label='Validation Loss')
	plt.legend(loc="upper right")
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.title('Loss Curve')
	plt.savefig('mortality Loss Curve')

	# Plot of Accuracy Curve
	plt.figure()
	plt.plot(epochs_acc, train_accuracies, label='Training Accuracy')
	plt.plot(epochs_acc, valid_accuracies, label='Validation Accuracy')
	plt.legend(loc="upper left")
	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Curve')
	plt.savefig('mortality Accuracy Curve')

	


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    y_true = pd.Series([y[0] for y in results], name='True')
    y_pred = pd.Series([y[1] for y in results], name='Predicted')
    df_confusion = pd.crosstab(y_true, y_pred)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    # print(df_confusion)
    # print(df_conf_norm)

    # Now Plot the Confusion Matrix
    plt.figure(figsize=(30,10))
    plt.tight_layout()
    plt.matshow(df_conf_norm, cmap='Blues') # imshow
    plt.title('Normalized Confusion Matrix', y=1.3)
    plt.colorbar()
    tick_marks = np.arange(len(df_conf_norm.columns))
    plt.xticks(tick_marks, class_names, verticalalignment='bottom', rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel(df_conf_norm.index.name)
    plt.xlabel(df_conf_norm.columns.name)

    # Show Values: https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
    for (i, j), z in np.ndenumerate(df_conf_norm):
        plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    plt.savefig('mortality Confusion Matrix', bbox_inches = "tight")

