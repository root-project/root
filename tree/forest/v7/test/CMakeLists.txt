# @author Jakob Blomer CERN

ROOT_STANDARD_LIBRARY_PACKAGE(CustomStruct
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/CustomStruct.hxx
                              SOURCES CustomStruct.cxx
                              LINKDEF CustomStructLinkDef.h
                              DEPENDENCIES RIO)
ROOT_ADD_GTEST(forest forest.cxx LIBRARIES ROOTDataFrame ROOTForest MathCore CustomStruct)
