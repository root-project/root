
from libROOTPython import AddBranchAttrSyntax, SetBranchAddressPyz

from ROOT import pythonization

from cppyy.gbl import TClass

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

    to_pythonize = [ 'TTree', 'TChain', 'TNtuple', 'TNtupleD' ]
    if name in to_pythonize:
        # Pythonizations that are common to TTree and its subclasses.
        # To avoid duplicating the same logic in the pythonizors of
        # the subclasses, inject the pythonizations for all the target
        # classes here.

        # Pythonic iterator
        klass.__iter__ = _TTree__iter__

        # tree.branch syntax
        AddBranchAttrSyntax(klass)

        # SetBranchAddress
        klass.SetBranchAddress = _SetBranchAddress

    return True
