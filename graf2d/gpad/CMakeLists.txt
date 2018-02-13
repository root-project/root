############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

if(root7)
    ROOT_GLOB_HEADERS(Gpad_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)

    ROOT_GENERATE_DICTIONARY(G__Gpadv7
                             ${Gpad_v7_dict_headers}
                             MODULE ROOTv7GraphicsGPad
                             OPTIONS "-writeEmptyRootPCM"
                             LINKDEF LinkDefv7.h)
    ROOT_OBJECT_LIBRARY(ROOTv7GraphicsGPad
                        ${root7src}
                        G__Gpadv7.cxx)

endif()

ROOT_GLOB_SOURCES(root6src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cxx)
ROOT_STANDARD_LIBRARY_PACKAGE(Gpad
                              HEADERS *.h
                              SOURCES ${root6src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Graf Hist)

if(root7)
    ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
