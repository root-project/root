############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpadv7 package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_HEADERS(Gpad_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/R*.hxx)
ROOT_GLOB_SOURCES(src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGpadv7
                              HEADERS ${Gpad_v7_dict_headers}
                              SOURCES ${src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES ROOTGraphicsPrimitives)

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
