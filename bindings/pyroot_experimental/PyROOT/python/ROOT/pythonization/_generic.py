from libROOTPython import AddPrettyPrintingPyz
from ROOT import pythonization

@pythonization
def pythonizegeneric(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add pretty printing via setting the __str__ special function
    AddPrettyPrintingPyz(klass)

    return True
