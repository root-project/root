# Author: Giacomo Parolini

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization

@pythonization("RDisplay", ns="ROOT::RDF")
def pythonize_rdisplay(klass):
    """
    Parameters:
    klass: class to be pythonized
    """
    def repr(klass):
        import ROOT
        opts = ROOT.RDF.RDisplay.RPrintOptions()
        opts.fFormat = ROOT.RDF.RDisplay.EPrintFormat.kHtml
        return klass.AsString(opts)
    klass._repr_html_ = repr
    klass.__repr__ = klass.AsString
