############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_HEADERS(headers inc/*.h)
ROOT_GLOB_SOURCES(sources src/*.cxx)

ROOT_STANDARD_LIBRARY_PACKAGE(RHTTPSniff
                              HEADERS ${headers}
                              SOURCES ${sources}
                              LIBRARIES ${SYSLIBS} ${CMAKE_DL_LIBS}
                              DEPENDENCIES RHTTP Thread Tree Hist Gpad Graf RIO XMLIO)
