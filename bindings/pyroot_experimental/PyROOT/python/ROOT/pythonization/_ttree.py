
from libROOTPython import AddBranchAttrSyntax, SetBranchAddressPyz

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

def _SetBranchAddress(self, *args):
    # Modify the behaviour if args is (const char*, void*)
    res = SetBranchAddressPyz(self, *args)

    if res is None:
        # Fall back to the original implementation for the rest of overloads
        res = self._OriginalSetBranchAddress(*args)
    
    return res

@pythonization
def pythonize_ttree(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TTree':
        # Pythonic iterator
        klass.__iter__ = _TTree__iter__

        # tree.branch syntax
        AddBranchAttrSyntax(klass)

        # SetBranchAddress
        klass._OriginalSetBranchAddress = klass.SetBranchAddress
        klass.SetBranchAddress = _SetBranchAddress

    return True
