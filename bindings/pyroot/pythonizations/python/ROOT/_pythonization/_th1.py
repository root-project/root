# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization


# Multiplication by constant

def _imul(self, c):
    # Parameters:
    # - self: histogram
    # - c: constant by which to multiply the histogram
    # Returns:
    # - A multiplied histogram (in place)
    self.Scale(c)
    return self


def _TH1_Constructor(self, *args, **kwargs):
    """
    Forward the arguments to the C++ constructor and give up ownership if the
    TH1 is attached to a TFile, which is the owner in that case.
    """
    import ROOT

    self._cpp_constructor(*args, **kwargs)
    tdir = self.GetDirectory()
    if tdir and type(tdir).__cpp_name__ == "TFile":
        ROOT.SetOwnership(self, False)

# The constructors need to be pythonized for each derived class separately:

@pythonization('TH1D')
def pythonize_th1(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TH1_Constructor

@pythonization('TH1F')
def pythonize_th1(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TH1_Constructor

@pythonization('THDF')
def pythonize_th1(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TH1_Constructor

@pythonization('TH2F')
def pythonize_th1(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TH1_Constructor

@pythonization('TProfile')
def pythonize_th1(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TH1_Constructor

@pythonization('TH1')
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized

    # Support hist *= scalar
    klass.__imul__ = _imul
