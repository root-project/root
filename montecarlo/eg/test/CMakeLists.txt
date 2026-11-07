# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

#### C++ TESTS ####

if(imt)
  set(extra_libraries Imt)
endif()
ROOT_ADD_GTEST(pdgdatabase pdg.cxx LIBRARIES EG ${extra_libraries})
