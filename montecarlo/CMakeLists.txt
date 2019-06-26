# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(eg)
if(pythia6)
  add_subdirectory(pythia6)
endif()
if(pythia8)
  add_subdirectory(pythia8)
endif()
if(vmc)
  add_subdirectory(vmc)
endif()
