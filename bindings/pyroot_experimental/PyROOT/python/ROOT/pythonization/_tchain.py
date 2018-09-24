
from ROOT import pythonization

from ._ttree import _SetBranchAddress as TTreeSetBranchAddress

@pythonization
def pythonize_tchain(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TChain':
        # SetBranchAddress
        # TChain overrides TTree's SetBranchAddress, so set it again (the Python method only forwards
        # onto a TTree*, so the C++ virtual function call will make sure the right method is used)
        klass._OriginalSetBranchAddress = klass.SetBranchAddress
        klass.SetBranchAddress = TTreeSetBranchAddress

    return True
