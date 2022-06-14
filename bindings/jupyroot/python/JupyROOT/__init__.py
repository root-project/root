#-----------------------------------------------------------------------------
#  Author: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#  Author: Enric Tejedor <enric.tejedor.saavedra@cern.ch> CERN
#-----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import sys
from JupyROOT.helpers import utils
if not 'win32' in sys.platform:
    from JupyROOT.helpers import cppcompleter

# Check if we are in the IPython shell
try:
    import builtins
except ImportError:
    import __builtin__ as builtins # Py2
_is_ipython = hasattr(builtins, '__IPYTHON__')

if _is_ipython:
    if not 'win32' in sys.platform:
        from IPython import get_ipython
        cppcompleter.load_ipython_extension(get_ipython())
    utils.iPythonize()
