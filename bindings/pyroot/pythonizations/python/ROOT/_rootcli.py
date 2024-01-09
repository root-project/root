import os, sys, subprocess

MYHOME = os.path.dirname(__file__)

def main():
    os.environ['LD_LIBRARY_PATH'] = os.path.join(MYHOME, 'lib')
    rootexe = os.path.join(MYHOME, 'bin', 'root.exe')

    return subprocess.call(rootexe)


if __name__ == "__main__":
    sys.exit(main())
