# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021


ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofie
  # Only the headers listed below are copied to the build tree and installed.
  # The ROperator*.hxx headers are an implementation detail of the model
  # generation and stay private to this library and ROOTTMVASofieParser, which
  # resolve them via the public build-interface include directory of this
  # target.
  NO_GLOB_HEADERS
  HEADERS
   TMVA/RModel.hxx
   TMVA/SOFIE_common.hxx
  SOURCES
    src/RModel.cxx
    src/SOFIE_common.cxx
    src/SOFIE_common_helpers.cxx
  DEPENDENCIES
    Core
)

# Tests are not enabled unconditionally because they require BLAS
if(test_tmva_sofie)
  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
