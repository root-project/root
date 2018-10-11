############################################################################
# CMakeLists.txt file for building ROOT hist/histdraw package
############################################################################

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/RHistDrawable.cxx)
    ROOT_GLOB_HEADERS(HistDraw_v7_dict_headers
                      v7/inc/ROOT/RHistDrawable.hxx
                      v7/inc/ROOT/RHistDrawingOpts.hxx
                     )

    ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHistDraw
         HEADERS ${HistDraw_v7_dict_headers}
         SOURCES *.cxx ${root7src}
         DICTIONARY_OPTIONS "-writeEmptyRootPCM"
         DEPENDENCIES ROOTGraphicsPrimitives)

    ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
