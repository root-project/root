# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(tmva)
if(TARGET Gui AND TARGET TreeViewer)
   add_subdirectory(tmvagui)
endif()

add_subdirectory(sofie)
add_subdirectory(sofie_parsers)
