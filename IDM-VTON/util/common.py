import platform, os

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')