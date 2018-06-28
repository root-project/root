
from libROOTPython import PythonizeTTree

from ROOT import pythonization

# TTree iterator
def _TTree__iter__(self):
    i = 0
    bytes_read = self.GetEntry(i)
    while 0 < bytes_read:
        yield self
        i += 1
        bytes_read = self.GetEntry(i)

    if bytes_read == -1:
        raise RuntimeError("TTree I/O error")

# Pythonizor function
@pythonization
def pythonize_ttree(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TTree':
        # Pythonic iterator
        klass.__iter__ = _TTree__iter__

        # C++ pythonizations
        # - tree.branch syntax
        PythonizeTTree(klass)

    return True
