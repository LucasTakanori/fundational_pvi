import os
from pathlib import Path

def rename_checkpoints(directory, old_string, new_string):
    files = os.listdir(directory)
    extension = '.pth'
    matching_files = [f for f in files if f.endswith(extension) and old_string in f]

    print(f"Set directory to '{directory}'...")
    if not matching_files:
        print(f"Found no files with extension '{extension}' and containing string '{old_string}'.")
    else:
        for k, filename in enumerate(matching_files):
            old_path = os.path.join(directory, filename)
            new_filename = filename.replace(old_string, new_string)
            new_path = os.path.join(directory, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed ({k + 1}/{len(matching_files)}): {filename} -> {new_filename}")


if __name__ == "__main__":
    wd = Path(r"D:\PviProject\artifacts\_final\s10-cnn-img-to-fiducials\checkpoints")
    rename_checkpoints(wd, "_best.pth", ".pth")