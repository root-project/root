# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_UNITTEST_DIR(ROOTHistDraw ROOTGpadv7)
endif()
