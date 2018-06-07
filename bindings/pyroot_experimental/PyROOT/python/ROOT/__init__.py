
import cppyy
import ROOT.pythonization as pyz

import pkgutil
import importlib

# Add pythonizations
for _, module_name, _ in  pkgutil.walk_packages(pyz.__path__):
  module = importlib.import_module(pyz.__name__ + '.' + module_name)
  cppyy.py.add_pythonization(module.get_pythonizor())

# Redirect ROOT to cppyy.gbl
import sys
sys.modules['ROOT'] = cppyy.gbl

