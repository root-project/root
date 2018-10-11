
import cppyy
import ROOT.pythonization as pyz

import pkgutil
import importlib

def pythonization(fn):
    """
    Pythonizor decorator to be used in pythonization modules.

    Parameters
    ----------
    fn : function
        Function that implements some pythonization.
        The function must accept two parameters: the class
        to be pythonized and the name of that class.
    """
    cppyy.py.add_pythonization(fn)

# Trigger the addition of the pythonizations
for _, module_name, _ in  pkgutil.walk_packages(pyz.__path__):
    module = importlib.import_module(pyz.__name__ + '.' + module_name)

# Redirect ROOT to cppyy.gbl
import sys
sys.modules['ROOT'] = cppyy.gbl
