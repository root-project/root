
from libROOTPython import AddBranchAttrSyntax, SetBranchAddressPyz, BranchPyz

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

def _Branch(self, *args):
    # Modify the behaviour if args is one of:
    # ( const char*, void*, const char*, Int_t = 32000 )
    # ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
    # ( const char*, T**, Int_t = 32000, Int_t = 99 )
    res = BranchPyz(self, *args)

    if res is None:
        # Fall back to the original implementation for the rest of overloads
        res = self._OriginalBranch(*args)

    return res

@pythonization
def pythonize_ttree(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    to_pythonize = [ 'TTree', 'TChain' ]
    if name in to_pythonize:
        # Pythonizations that are common to TTree and its subclasses.
        # To avoid duplicating the same logic in the pythonizors of
        # the subclasses, inject the pythonizations for all the target
        # classes here.
        # TChain needs to be explicitly pythonized because it redefines
        # SetBranchAddress in C++. As a consequence, TChain does not
        # inherit TTree's pythonization for SetBranchAddress, which
        # needs to be injected to TChain too. This is not the case for
        # other classes like TNtuple, which will inherit all the
        # pythonizations added here for TTree.

        # Pythonic iterator
        klass.__iter__ = _TTree__iter__

        # tree.branch syntax
        AddBranchAttrSyntax(klass)

        # SetBranchAddress
        klass._OriginalSetBranchAddress = klass.SetBranchAddress
        klass.SetBranchAddress = _SetBranchAddress

        # Branch
        klass._OriginalBranch = klass.Branch
        klass.Branch = _Branch

    return True
