############################################################################
# CMakeLists.txt file for building ROOT hist/histdraw package
############################################################################

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)

    ROOT_STANDARD_LIBRARY_PACKAGE(HistDraw
         SOURCES *.cxx ${root7src}
         DICTIONARY_OPTIONS "-writeEmptyRootPCM"
         DEPENDENCIES Gpad Hist)
endif()

