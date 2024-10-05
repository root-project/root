# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.


# RooAbsCollection and subclasses pythonizations
if(NOT MSVC OR CMAKE_SIZEOF_VOID_P EQUAL 4 OR win_broken_tests)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_rooabscollection rooabscollection.py)
endif()
ROOT_ADD_PYUNITTEST(pyroot_roofit_rooarglist rooarglist.py)

# RooDataHist pythonisations
ROOT_ADD_PYUNITTEST(pyroot_roofit_roodatahist_ploton roodatahist_ploton.py)

# RooDataSet pythonisations
ROOT_ADD_PYUNITTEST(pyroot_roofit_roodataset roodataset.py)

# RooWorkspace pythonizations
ROOT_ADD_PYUNITTEST(pyroot_roofit_rooabspdf_fitto rooabspdf_fitto.py)
ROOT_ADD_PYUNITTEST(pyroot_roofit_rooabsreal_ploton rooabsreal_ploton.py)

ROOT_ADD_PYUNITTEST(pyroot_roofit_roolinkedlist roolinkedlist.py)

if(NOT MSVC OR win_broken_tests)
  # Test pythonizations for the RooFitHS3 package, which is not built on Windows.
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roojsonfactorywstool roojsonfactorywstool.py)

  # Other pythonizations that fail on Windows for unknown reasons
  ROOT_ADD_PYUNITTEST(pyroot_roofit_rooglobalfunc rooglobalfunc.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roosimultaneous roosimultaneous.py)

  # RooWorkspace pythonization that fails on Windows
  ROOT_ADD_PYUNITTEST(pyroot_roofit_rooworkspace rooworkspace.py)
endif()

# NumPy compatibility
ROOT_ADD_PYUNITTEST(pyroot_roofit_roodataset_numpy roodataset_numpy.py PYTHON_DEPS numpy)
ROOT_ADD_PYUNITTEST(pyroot_roofit_roodatahist_numpy roodatahist_numpy.py PYTHON_DEPS numpy)
