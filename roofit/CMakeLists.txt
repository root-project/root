# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# If `roofit-arch-libs` is set, the computation libraries of RooFit are compiled multiple times
# to match target architectures that support vector instructions. Windows platform and ICC compiler
# need special code and testing, thus the feature has not been implemented yet for these.
# The option is used in roofit/roofit/CMakeLists.txt
if(ROOT_PLATFORM MATCHES "linux|macosx" AND CMAKE_SYSTEM_PROCESSOR MATCHES x86_64 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(roofit-arch-libs ON)
endif()

add_subdirectory(roofitcore)
add_subdirectory(roofit)
if(mathmore)
  add_subdirectory(roofitmore)
endif()
add_subdirectory(roostats)
if(xml)
  add_subdirectory(histfactory)
endif()
