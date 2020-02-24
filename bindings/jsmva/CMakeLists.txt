# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

####################################################################################
# CMakeLists.txt file for building JsMVA bindings/pyroot_experimental/PyROOT package
####################################################################################

set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PYROOTDIR})
set(JsMVADirName python/JsMVA)

file(COPY ${JsMVADirName} DESTINATION ${localruntimedir})
install(DIRECTORY ${JsMVADirName} DESTINATION ${CMAKE_INSTALL_LIBDIR})
