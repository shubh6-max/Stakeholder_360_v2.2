import os

def print_folder_structure(folder_path, indent=0, exclude=None):
    if exclude is None:
        exclude = []

    try:
        items = os.listdir(folder_path)
    except PermissionError:
        print(" " * indent + "🚫 Permission Denied")
        return

    for item in items:
        if item in exclude:   # skip excluded folders
            continue
        item_path = os.path.join(folder_path, item)
        print(" " * indent + "├── " + item)
        if os.path.isdir(item_path):
            print_folder_structure(item_path, indent + 4, exclude)

if __name__ == "__main__":
    folder = r"C:\Users\ShubhamVishwasPurani\OneDrive - TheMathCompany Private Limited\Desktop\Stakeholder_360"
    
    if os.path.exists(folder) and os.path.isdir(folder):
        print(folder)
        print_folder_structure(folder, exclude=["venv"])
    else:
        print("❌ Invalid folder path.")
