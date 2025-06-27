# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(hist)
add_subdirectory(histpainter)
if(root7)
  add_subdirectory(histv7)
endif()
if (spectrum)
   add_subdirectory(spectrum)
   add_subdirectory(spectrumpainter)
endif()
if(unfold)
  add_subdirectory(unfold)
endif()
if(CMAKE_Fortran_COMPILER)
  add_subdirectory(hbook)
endif()
