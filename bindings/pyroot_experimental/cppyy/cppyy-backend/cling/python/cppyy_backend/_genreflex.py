from __future__ import print_function
import os, sys, subprocess

MYHOME = os.path.dirname(__file__)

def main():
    if len(sys.argv) == 2 and \
           (sys.argv[1] == '--cflags' or sys.argv[1] == '--cppflags'):
        print('-I%s/include' % (MYHOME,))
        return 0
    os.environ['LD_LIBRARY_PATH'] = os.path.join(MYHOME, 'lib')
    genreflex = os.path.join(MYHOME, 'bin', 'genreflex')
    if not os.path.exists(genreflex):
        raise RuntimeError("genreflex not installed in standard location")
    return subprocess.call([genreflex] + sys.argv[1:])
