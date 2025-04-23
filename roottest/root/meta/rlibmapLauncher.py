'''
To cope with the fact that rlibmap exists with CMake builds and not Makefile ones.
'''

import sys
import subprocess

def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


errmsg="ERROR: The rlibmap utility is not available in ROOT6. In order to produce rootmap files you can use the genreflex utils (genreflex -h for more information) or the rootcling utility (rootcling -h for more information)."

rlibmapName = "rlibmap"

if which(rlibmapName):
   sys.exit(subprocess.call(rlibmapName, shell=True))
else:
   # Fake rlibmap
   print(errmsg)
   sys.exit(1)
