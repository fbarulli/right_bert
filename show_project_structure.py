import os

def display_project_structure(root_path, indent=""):
    """
    Recursively displays the structure of a project, starting from a given root path.

    Args:
        root_path: The path to the root directory of the project.
        indent:  A string representing the current indentation level (used for recursive calls).
    """

    try:
        items = os.listdir(root_path)
    except OSError as e:
        print(f"{indent}ERROR: Could not access {root_path}: {e}")
        return

    for item in items:
        item_path = os.path.join(root_path, item)

        if os.path.isfile(item_path):
            print(f"{indent}- {item}")  # File
        elif os.path.isdir(item_path):
            print(f"{indent}+ {item}/")  # Directory
            display_project_structure(item_path, indent + "  ")  # Recursive call for subdirectories
        else:
            print(f"{indent}? {item} (Unknown type)") #Handles other types like symbolic links

def main():
    """
    Gets the project root directory from the user and displays the structure.
    """

    # Get project root.  Handle cases where the user provides a relative or invalid path.
    while True:
        project_root = input("Enter the absolute path to the project root directory: ")
        if os.path.isdir(project_root):
            break
        else:
            print("Invalid directory path. Please enter a valid path.")

    print("\nProject Structure:")
    display_project_structure(project_root)


if __name__ == "__main__":
    main()