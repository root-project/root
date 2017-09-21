############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

if(root7)
    set(root7src v7/src/)
    ROOT_GLOB_HEADERS(Gpad_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Gpad
                              HEADERS *.h ${Gpad_v7_dict_headers}
                              SOURCES *.cxx ${root7src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Graf Hist)

if(testing)
    if(root7)
        add_subdirectory(v7/test)
    endif()
endif()