""" cppyy_backend loader
"""

import os, ctypes

def load_cpp_backend():
    try:
      # normal load, allowing for user overrides of LD_LIBRARY_PATH
        c = ctypes.CDLL("libcppyy_backend.so", ctypes.RTLD_GLOBAL)
    except OSError:
      # failed ... load dependencies explicitly
        libpath = os.path.join(os.path.dirname(__file__), 'lib')
        for dep in ['libCore.so', 'libThread.so', 'libRIO.so', 'libCling.so']:
            ctypes.CDLL(os.path.join(libpath, dep), ctypes.RTLD_GLOBAL)
        c = ctypes.CDLL(os.path.join(libpath, 'libcppyy_backend.so'), ctypes.RTLD_GLOBAL)

    return c
