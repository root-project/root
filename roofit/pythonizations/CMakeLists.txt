# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

################################################################
# CMakeLists.txt file for building ROOT pythonizations libraries
################################################################

set(py_sources
  __init__.py
  _rooabscollection.py
  _rooabsdata.py
  _rooabspdf.py
  _rooabsreallvalue.py
  _rooabsreal.py
  _rooarglist.py
  _rooargset.py
  _roocategory.py
  _roochi2var.py
  _roodatahist.py
  _roodataset.py
  _roodecays.py
  _roogenfitstudy.py
  _rooglobalfunc.py
  _roojsonfactorywstool.py
  _roomcstudy.py
  _roomsgservice.py
  _roonllvar.py
  _rooprodpdf.py
  _roorealvar.py
  _roosimultaneous.py
  _roosimwstool.py
  _roostats.py
  _roovectordatastore.py
  _rooworkspace.py
  _utils.py
)

set(pyz_libname "roofit_pythonizations")

foreach(py_source ${py_sources})
  set(py_source_full "python/${pyz_libname}/${py_source}")
  configure_file(${py_source_full} "${localruntimedir}/ROOT/_${pyz_libname}/${py_source}" COPYONLY)
  install(FILES ${py_source_full} DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/ROOT/_${pyz_libname}/ COMPONENT libraries)
endforeach()

ROOT_ADD_TEST_SUBDIRECTORY(test)
