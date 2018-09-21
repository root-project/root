
from ROOT import pythonization

from ._ttree import _SetBranchAddress as TTreeSetBranchAddress

# Pythonizor function
@pythonization
def pythonize_tchain(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TChain':
        # SetBranchAddress
        klass._OriginalSetBranchAddress = klass.SetBranchAddress
        klass.SetBranchAddress = TTreeSetBranchAddress

    return True
