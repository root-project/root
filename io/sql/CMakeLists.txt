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
