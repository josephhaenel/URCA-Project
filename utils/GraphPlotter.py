import os
import matplotlib.pyplot as plt

def save_plots(history, output_dir):
    """
    Save plots for all metrics in the training history.
    """
    num_metrics = len(history.history) // 2  # Assuming there's a 'val_' version for each metric
    plt.figure(figsize=(15, num_metrics * 5))

    for i, key in enumerate(history.history.keys()):
        if not key.startswith('val_'):  # Avoid plotting validation metrics separately
            # Create subplot for each metric
            plt.subplot(num_metrics, 1, i // 2 + 1)
            plt.plot(history.history[key], label=f'Training {key}')
            val_key = f'val_{key}'
            if val_key in history.history:
                plt.plot(history.history[val_key], label=f'Validation {key}')
            plt.title(key.capitalize())
            plt.xlabel('Epochs')
            plt.ylabel(key)
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics_plots.png'))
    plt.show()

def save_history_to_txt(history, output_dir):
    """
    Save training history data to a text file.
    """
    history_file_path = os.path.join(output_dir, 'training_history.txt')
    with open(history_file_path, 'w') as file:
        for key in history.history.keys():
            file.write(f"{key}\n")
            for value in history.history[key]:
                file.write(f"{value}\n")
            file.write("\n")
