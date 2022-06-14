# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(mathcore)
if(mathmore)
  add_subdirectory(mathmore)
endif()
add_subdirectory(matrix)
add_subdirectory(minuit)
if(minuit2)
  add_subdirectory(minuit2)
endif()
add_subdirectory(fumili)
add_subdirectory(physics)
if(mlp)
   add_subdirectory(mlp)
endif(mlp)
add_subdirectory(quadp)
add_subdirectory(foam)
add_subdirectory(smatrix)
add_subdirectory(splot)
#add_subdirectory(mathcore/test EXCLUDE_FROM_ALL)
add_subdirectory(genvector)
if(tmva)
  add_subdirectory(genetic)
endif()
if(unuran)
  add_subdirectory(unuran)
endif()
if(fftw3)
  add_subdirectory(fftw)
endif()

if(r)
  add_subdirectory(rtools)
endif()

add_subdirectory(vecops)
