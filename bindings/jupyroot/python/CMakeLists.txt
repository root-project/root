# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building JupyROOT
###########################################################

ROOT_PYTHON_PACKAGE(JupyROOT
  SOURCES
  JupyROOT/__init__.py
  JupyROOT/helpers/__init__.py
  JupyROOT/helpers/cppcompleter.py
  JupyROOT/helpers/handlers.py
  JupyROOT/helpers/utils.py
  JupyROOT/html/__init__.py
  JupyROOT/html/cpphighlighter.py
  JupyROOT/kernel/__init__.py
  JupyROOT/kernel/rootkernel.py
  JupyROOT/kernel/utils.py
  JupyROOT/kernel/magics/__init__.py
  JupyROOT/kernel/magics/cppmagic.py
  JupyROOT/kernel/magics/jsrootmagic.py
  JupyROOT/magics/__init__.py
  JupyROOT/magics/cppmagic.py
  JupyROOT/magics/jsrootmagic.py
)
