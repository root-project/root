# Author: Pawan Johnson, Vincenzo Eduardo Padulano, Enric Tejedor  CERN  07/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
"""
Contains constants needed for _rdf_pyz to convert datatypes for numba declarable types.
It is in a separate module so as to avoid a numpy dependency for ROOT.
"""
try:
    import numpy
except:
    raise ImportError("Failed to import numpy during call to determine function signature.")

FUNDAMENTAL_PYTHON_TYPES = {
    bool: 'bool',
    int : 'int',
    float: 'double',
}

TREE_TO_NUMBA = {
    'C': 'str', 
    'Char_t': 'int',
    'UChat_t': 'unsigned int',
    'Short_t': 'int',
    'UShort_t': 'unsigned int',
    'Int_t': 'int',
    'UInt_t': 'unsigned int',
    'Float_t': 'float',
    'Float16_t': 'float',
    'Double_t': 'double',
    'Double32_t': 'double',
    'Long64_t': 'long',
    'ULong64_t': 'unsigned long',
    'Long_t': 'long',
    'ULong_t': 'unsigned long',
    'Bool_t': 'bool',
}

NUMPY_TO_TREE = {
    numpy.bool_ : 'Bool_t',
    numpy.byte :  'Char_t',
    numpy.ubyte : 'UChar_t',
    numpy.short : 'Short_t',
    numpy.ushort : 'UShort_t',
    numpy.intc : 'Int_t',
    numpy.uintc : 'UInt_t',
    numpy.int_ : 'Long_t',
    numpy.uint : 'ULong_t',
    numpy.half : 'Float16_t',
    numpy.float16: 'Float16_t',
    numpy.single : 'Float_t',
    numpy.double : 'Double_t',
    numpy.intp : 'Int_t',
    numpy.int8: 'Int_t',
    numpy.int16 : 'Int_t',
    numpy.int32: 'Int_t',
    numpy.int64 : 'Long_t',
    numpy.float_ : 'Float_t',
    numpy.float32 : 'Float_t',
    numpy.float64 : 'Double_t',
}

