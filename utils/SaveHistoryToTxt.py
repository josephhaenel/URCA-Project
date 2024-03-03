import os

def save_history_to_txt(history, output_dir):
    """
    Save training history data to a text file.
    """
    history_file_path = os.path.join(output_dir, 'training_history.txt')
    with open(history_file_path, 'w') as file:
        for key in history.history.keys():
            file.write(f"{key}\n")
            for value in history.history[key]:

                if hasattr(value, 'numpy'):
                    value = value.numpy()
                file.write(f"{value}\n")
            file.write("\n")

