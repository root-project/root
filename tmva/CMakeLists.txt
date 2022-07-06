# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

find_package(LAPACK)

add_subdirectory(tmva)
add_subdirectory(tmvagui)

# Build pyMVA. Allows users to call python mva's from c++ ROOT/TMVA.
if(tmva-pymva AND NUMPY_FOUND)
   add_subdirectory(pymva)
endif()

# Build RMVA. Allows users to call R mva's from c++ ROOT/TMVA.
if(r OR tmva-rmva)
   add_subdirectory(rmva)
endif()

add_subdirectory(sofie)
#parser depends on protobuf 
#found if flag tmva-sofie is on
if (tmva-sofie)
   add_subdirectory(sofie_parsers)
endif()
