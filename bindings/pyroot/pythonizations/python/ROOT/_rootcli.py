import os
import subprocess
import sys

ROOT_HOME = os.path.dirname(__file__)


def main() -> None:
    # Ensure all ROOT libraries are found at runtime by the process
    os.environ['LD_LIBRARY_PATH'] = os.path.join(ROOT_HOME, 'lib')
    # Finds the ROOT executable from the current installation directory
    bindir = os.path.join(ROOT_HOME, 'bin')
    rootexe = os.path.join(bindir, 'root.exe')
    if not os.path.exists(rootexe):
        raise FileNotFoundError(
            f"Could not find 'root' executable in directory '{bindir}'. "
            "Something is wrong in the ROOT installation.")
    # Make sure command line arguments are preserved
    args = [rootexe] + sys.argv[1:]
    # Run the actual ROOT executable and return the exit code to the main Python process
    out = subprocess.run(args)
    return out.returncode


if __name__ == "__main__":
    sys.exit(main())
