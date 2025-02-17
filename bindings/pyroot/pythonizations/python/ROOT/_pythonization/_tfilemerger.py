# Author: Giacomo Parolini CERN  12/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization

def _TFileMergerExit(obj, exc_type, exc_val, exc_tb):
    """
    Close the merger's output file.
    Signature and return value are imposed by Python, see
    https://docs.python.org/3/library/stdtypes.html#typecontextmanager
    """
    obj.CloseOutputFile()
    return False


@pythonization('TFileMerger')
def pythonize_tfile_merger(klass):
    """
    TFileMerger works as a context manager.
    """
    # Pythonization for __enter__ and __exit__ methods
    # These make TFileMerger usable in a `with` statement as a context manager
    klass.__enter__ = lambda merger: merger
    klass.__exit__ = _TFileMergerExit
