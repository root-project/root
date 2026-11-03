# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/unuran package
############################################################################

set(UNURAN_HEADERS
  TUnuran.h
  TUnuranBaseDist.h
  TUnuranContDist.h
  TUnuranDiscrDist.h
  TUnuranEmpDist.h
  TUnuranMultiContDist.h
  TUnuranSampler.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Unuran
  HEADERS
    ${UNURAN_HEADERS}
  SOURCES
    src/TUnuran.cxx
    src/TUnuranContDist.cxx
    src/TUnuranDiscrDist.cxx
    src/TUnuranEmpDist.cxx
    src/TUnuranMultiContDist.cxx
    src/TUnuranSampler.cxx
  LIBRARIES
    Unuran::unuran
  DEPENDENCIES
    Core
    Hist
    MathCore
)

set(RELATIVE_INC_HEADERS ${UNURAN_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Unuran
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/UnuranDistrAdapter.h
    src/UnuranRng.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
