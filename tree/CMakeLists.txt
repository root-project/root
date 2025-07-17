# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(tree)
add_subdirectory(treeplayer)
add_subdirectory(treeviewer)
if(webgui)
  add_subdirectory(webviewer)
endif()
add_subdirectory(dataframe)
add_subdirectory(ntuple)
add_subdirectory(ntuplebrowse)
add_subdirectory(ntupleutil)
add_subdirectory(readspeed)
