# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_library(cppyy_backend STATIC clingwrapper/src/clingwrapper.cxx)
target_link_libraries(cppyy_backend Core)
target_compile_options(cppyy_backend PRIVATE -fPIC)
