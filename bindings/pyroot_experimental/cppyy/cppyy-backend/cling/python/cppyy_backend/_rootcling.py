import os, sys, subprocess

MYHOME = os.path.dirname(__file__)

def main():
    os.environ['LD_LIBRARY_PATH'] = os.path.join(MYHOME, 'lib')
    rootcling = os.path.join(MYHOME, 'bin', 'rootcling')
    if not os.path.exists(rootcling):
        raise RuntimeError("rootcling not installed in standard location")
    return subprocess.call([rootcling] + sys.argv[1:])
