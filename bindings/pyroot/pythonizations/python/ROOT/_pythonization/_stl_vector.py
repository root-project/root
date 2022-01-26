# Author: Stefan Wunsch, Enric Tejedor CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from ._rvec import add_array_interface_property

def _data_vec_char(self):
    # vector<char>::data() returns char*.
    # Cppyy attemps to convert char* into Python string, but if the
    # character sequence is not null-terminated the conversion fails.
    # This is likely to happen with the result of vector<char>::data().
    # For the conversion char* -> str to succeed when calling data(),
    # temporarily append a null character to the vector<char>.
    self.push_back('\0')
    d = self._original_data()
    self.pop_back()
    return d

@pythonization("vector<", ns="std", is_prefix=True)
def pythonize_stl_vector(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add numpy array interface
    # NOTE: The pythonization is reused from ROOT::VecOps::RVec
    add_array_interface_property(klass, name)

    # Inject custom vector<char>::data()
    if klass.value_type == 'char':
        klass._original_data = klass.data
        klass.data = _data_vec_char
