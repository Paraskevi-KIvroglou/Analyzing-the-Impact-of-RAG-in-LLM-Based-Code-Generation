import os
from pathlib import PurePath

def create_dir_in_path(new_directory_name):
    """ Create a directory in Python using a dir name."""
    # Get the current working directory
    current_directory = os.getcwd()
    print(current_directory)

    # Method 2: Use pathlib (recommended)
    path_obj = PurePath(current_directory)
    print(path_obj.parts)

    parts = list(path_obj.parts)
    if parts[0] == "C:\\":
        parts[0] = "C:"
    parts.append(str(new_directory_name))
    print(parts)

    full_path = " ".join(parts)
    print(full_path)
    # Create the full path for the new directory
    new_directory_path = os.path.join(full_path)

    # Create the directory
    os.makedirs(new_directory_path, exist_ok=True)

    print(f"Directory created at: {new_directory_path}")

    return parts