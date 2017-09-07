############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(HistPainter
                              SOURCES *.cxx ${root7src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES  Graf Hist Matrix MathCore Gpad)

