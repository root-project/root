from libROOTPython import PythonizeGeneric
from ROOT import pythonization

@pythonization
def pythonizegeneric(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add pythonizations generically to all classes
    PythonizeGeneric(klass)

    return True
