import os

def _create_directory(self, path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)