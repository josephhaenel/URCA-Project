import os

def _create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)