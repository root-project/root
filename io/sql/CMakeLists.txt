# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/sql package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(SQLIO
                              HEADERS
                                 TBufferSQL2.h
                                 TKeySQL.h
                                 TSQLClassInfo.h
                                 TSQLFile.h
                                 TSQLObjectData.h
                                 TSQLStructure.h
                              SOURCES
                                 src/TBufferSQL2.cxx
                                 src/TKeySQL.cxx
                                 src/TSQLClassInfo.cxx
                                 src/TSQLFile.cxx
                                 src/TSQLObjectData.cxx
                                 src/TSQLStructure.cxx
                              DEPENDENCIES Net RIO)
