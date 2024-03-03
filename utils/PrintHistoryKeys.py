import tensorflow as tf

def print_history_keys(self, history: tf.keras.callbacks.History) -> None:
    """
    Prints keys from the training history. For debugging purposes.

    Parameters:
        history (tf.keras.callbacks.History): The history object from model training.
    """
    print("Keys in training history:")
    for key in history.history.keys():
        print(key)