from ROOT import pythonization
from ROOT.pythonization._rvec import add_array_interface_property


@pythonization
def pythonize_stl_vector(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name.startswith("std::vector<"):
        # Add numpy array interface
        # NOTE: The pythonization is reused from ROOT::VecOps::RVec
        add_array_interface_property(klass, name)

    return True
