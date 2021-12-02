# Author: Enric Tejedor, Danilo Piparo CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from os import environ

# Prevent cppyy's check for the PCH
environ['CLING_STANDARD_PCH'] = 'none'

# Prevent cppyy's check for extra header directory
environ['CPPYY_API_PATH'] = 'none'

# Prevent cppyy from filtering ROOT libraries
environ['CPPYY_NO_ROOT_FILTER'] = '1'

import cppyy
if not 'ROOTSYS' in environ:
    # Revert setting made by cppyy
    cppyy.gbl.gROOT.SetBatch(False)

# import libROOTPythonizations with Python version number
import sys, importlib
major, minor = sys.version_info[0:2]
librootpyz_mod_name = 'libROOTPythonizations{}_{}'.format(major, minor)
importlib.import_module(librootpyz_mod_name)

# ensure 'import libROOTPythonizations' will find the versioned module
sys.modules['libROOTPythonizations'] = sys.modules[librootpyz_mod_name]

# Trigger the addition of the pythonizations
from ._pythonization import _register_pythonizations
_register_pythonizations()

# Check if we are in the IPython shell
if major == 3:
    import builtins
else:
    import __builtin__ as builtins  

_is_ipython = hasattr(builtins, '__IPYTHON__')

# Configure ROOT facade module
import sys
from ._facade import ROOTFacade
sys.modules[__name__] = ROOTFacade(sys.modules[__name__], _is_ipython)

# Configuration for usage from Jupyter notebooks
if _is_ipython:
    from IPython import get_ipython
    ip = get_ipython()
    if hasattr(ip,"kernel"):
        import JupyROOT
        import JsMVA

# Register cleanup
import atexit
def cleanup():
    # If spawned, stop thread which processes ROOT events
    facade = sys.modules[__name__]
    if 'app' in facade.__dict__ and hasattr(facade.__dict__['app'], 'process_root_events'):
        facade.__dict__['app'].keep_polling = False
        facade.__dict__['app'].process_root_events.join()

    if 'libROOTPythonizations' in sys.modules:
        backend = sys.modules['libROOTPythonizations']

        # Make sure all the objects regulated by PyROOT are deleted and their
        # Python proxies are properly nonified.
        backend.ClearProxiedObjects()

        from ROOT import PyConfig
        if PyConfig.ShutDown:
            # Hard teardown: run part of the gROOT shutdown sequence.
            # Running it here ensures that it is done before any ROOT libraries
            # are off-loaded, with unspecified order of static object destruction.
            backend.gROOT.EndOfProcessCleanups()

atexit.register(cleanup)
