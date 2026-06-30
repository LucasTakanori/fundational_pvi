from pathlib import Path

# def rename_files(directory: Path,
#                  extension: str,
#                  old: str,
#                  new: str = None,
#                  recursive: bool=False):
#
#     if recursive:
#         all_items = list(directory.rglob("*"))
#     else:
#         all_items = list(directory.iterdir())
#
#     matching_files = [item for item in all_items
#                       if item.is_file()
#                       and item.name.endswith(extension)
#                       and old in item.name]
#
#     new = '' if new is None else new
#
#     if not matching_files:
#         print(f"Found no files with extension '{extension}' and containing string '{old}'.")
#     else:
#         print(f"Set directory to: \n\t '{directory}'...")
#         for k, file_path in enumerate(matching_files):
#             filename = file_path.name
#             new_filename = filename.replace(old, new)
#             new_path = file_path.parent / new_filename
#
#             file_path.rename(new_path)
#             print(f"{k + 1}/{len(matching_files)} | {file_path.parent}':")
#             print(f"\t Rename '{filename}' -> '{new_filename}'")

def move_folders(source: Path,
                 destination: Path,
                 items: list[str]=None) -> None:
    all_dirs = [item for item in list(source.iterdir()) if item.is_dir()]
    if items is not None:
        matching_dirs = [d for d in all_dirs if d.name in items]
    else:
        matching_dirs = all_dirs

    if not matching_dirs:
        return

    destination.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    directory = Path(r"D:\PviProject\artifacts")
    rename_files(directory=directory,
                 extension="json",
                 old="_terminal.json",
                 new=".json",
                 recursive=True)