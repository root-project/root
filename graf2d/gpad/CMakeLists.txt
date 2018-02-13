############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_SOURCES(root6src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cxx)
ROOT_STANDARD_LIBRARY_PACKAGE(Gpad
                              HEADERS *.h
                              SOURCES ${root6src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Graf Hist)
