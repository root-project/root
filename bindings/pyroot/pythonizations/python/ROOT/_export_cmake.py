from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    # This trick is only necessary for the wheel distribution
    try:
        if "a" in version("ROOT"):
            root_site_package = Path(__file__).resolve().parent
        
            # Print a warning to stdout that eval will ignore but would be visible to user who accidentally runs raw command
            print('# ERROR: You should eval the output of this command, not run it directly in your shell.\n# Use eval "$(export_cmake)"')
            # Print the export commands to be eval'd by user's shell
            print(f'export CMAKE_PREFIX_PATH="{root_site_package}"')

    except PackageNotFoundError:
        print("Setting CMAKE_PREFIX_PATH is only necessary when linking against ROOT's PyPI wheels. Please use thisroot.sh.")

if __name__ == "__main__":
    main()
