# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(hist)             # special CMakeLists.txt
add_subdirectory(histpainter)      # special CMakeLists.txt
if(root7)
   add_subdirectory(histdraw)         # special CMakeLists.txt
endif()
if (spectrum)
   add_subdirectory(spectrum)
   add_subdirectory(spectrumpainter)  # special CMakeLists.txt
endif()
if(xml)
  add_subdirectory(unfold)
endif()
if(CMAKE_Fortran_COMPILER)
  add_subdirectory(hbook)
endif()
