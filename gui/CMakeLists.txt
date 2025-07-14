# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(gui)
add_subdirectory(ged)
add_subdirectory(fitpanel)
add_subdirectory(guibuilder)
add_subdirectory(guihtml)
add_subdirectory(recorder)

if(webgui)
   if(cefweb)
      add_subdirectory(cefdisplay)
   endif()
   if(qt6web)
      add_subdirectory(qt6webdisplay)
   endif()

   add_subdirectory(webdisplay)

   add_subdirectory(webgui6)

   add_subdirectory(browsable)
   add_subdirectory(browserv7)

   if(root7)
      add_subdirectory(canvaspainter)
      add_subdirectory(fitpanelv7)
   endif()
endif()
