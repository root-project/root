# Authors:
# * Jonas Rembser 09/2023

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


class RooLinkedList(object):

    def Add(self, arg):
        # The Add() method is only changed for the sake of changing the memory
        # policy of the cppyy overload. The signature of the original C++
        # function is RooLinkedList::Add(RooAbsArg *arg). PyROOT is wrongly
        # interpreting this as the RooLinkedList taking ownership of the
        # RooAbsArg. This results in a memory leak because nobody feels
        # responsible for deleting the arg. This can be fixed by setting the
        # memory policy of the method to "strict" and not to "heuristic".
        #
        # This might become unnecessary in the future if it is decided to set
        # the global memory policy to "strict" in _facade.py.

        import ROOT

        self._Add.__mempolicy__ = ROOT.kMemoryStrict
        return self._Add(arg)
