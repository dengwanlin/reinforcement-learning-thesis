import os
import sys

def print_directory_tree(path, prefix="", is_last=True):
    """
    Recursively print directory structure, supporting hidden files.
    """
    try:
        items = os.listdir(path)
    except PermissionError:
        print(f"{prefix}  can not be accessed:{path}as rights not enough")
        return
    except Exception as e:
        print(f"{prefix} error:{e}")
        return

    # sorting, to make the output better
    items.sort()

    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_dir = os.path.isdir(item_path)
        is_last_item = (i == len(items) - 1)

        # construct the connector symbol
        connector = "└── " if is_last_item else "├── "
        branch = "    " if is_last else "│   "

        # to print current item
        print(f"{prefix}{connector}{item}{'/' if is_dir else ''}")

        # Recursive processing subdirectories
        if is_dir:
            extension = "    " if is_last_item else "│   "
            print_directory_tree(item_path, prefix + extension, is_last_item)


def main():
    # Check if the path parameter was passed.
    if len(sys.argv) != 2:
        print(" Usage: python3 show_tree.py <target path>")
        print("Example: python3 show_tree.py /homes/sohawan2/.../seed0")
        sys.exit(1)

    target_path = sys.argv[1]

    # Check if the path exists
    if not os.path.exists(target_path):
        print(f"❌ Path does not exist:{target_path}")
        sys.exit(1)

    if not os.path.isdir(target_path):
        print(f"❌ Not a table of contents:{target_path}")
        sys.exit(1)

    # Print title
    print(f"📁 Directory structure:{target_path}")
    print("-" * 60)

    # Start printing
    print_directory_tree(target_path)


if __name__ == "__main__":
    main()
