# The scripts load the dictionary libraries by their file names.
if(MSVC)
  return()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libjet Jet_v1.h
                                    SELECTION inc/jet_selection.xml
                                    LIBNAME libjet
                                    NO_ROOTMAP
                                    OPTIONS --rootmap=libjet.rootmap --rootmap-lib=libjet.so)
set(fixtures ${GENERATE_REFLEX_TEST}-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libsjet SJet_v1.h
                                    SELECTION inc/sjet_selection.xml
                                    LIBNAME libsjet
                                    NO_ROOTMAP
                                    OPTIONS --rootmap=libsjet.rootmap --rootmap-lib=libsjet.so)
list(APPEND fixtures ${GENERATE_REFLEX_TEST}-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libbtag BTagVertex_v1.h
                                    SELECTION inc/btag_selection.xml
                                    LIBNAME libbtag
                                    NO_ROOTMAP
                                    OPTIONS --rootmap=libbtag.rootmap --rootmap-lib=libbtag.so)
list(APPEND fixtures ${GENERATE_REFLEX_TEST}-fixture)

foreach(case case1 case2 case3 case4)
  ROOTTEST_ADD_TEST(${case}
                    COMMAND ${ROOT_root_CMD} -l -b
                    INPUT scripts/${case}.script
                    OUTREF ${case}.ref
                    FIXTURES_REQUIRED ${fixtures})

  # Same, but with the rootmap files of all libraries loaded first.
  ROOTTEST_ADD_TEST(rmap-${case}
                    COMMAND ${ROOT_root_CMD} -l -b ${CMAKE_CURRENT_SOURCE_DIR}/scripts/loadrootmap.C
                    INPUT scripts/${case}.script
                    OUTREF rmap-${case}.ref
                    FIXTURES_REQUIRED ${fixtures})
endforeach()
