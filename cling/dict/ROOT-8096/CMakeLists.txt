# ROOT-8096

add_custom_command(OUTPUT dict.cxx
COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/definitions.h -o dict.cxx   --selection_file=${CMAKE_CURRENT_SOURCE_DIR}/selection.xml --rootmap libROOT-8096.rootmap --rootmap-lib libROOT-8096.so )

set_source_files_properties(dict.cxx PROPERTIES GENERATED TRUE)

add_library(ROOT-8096 EXCLUDE_FROM_ALL SHARED dict.cxx)
set_target_properties(ROOT-8096 PROPERTIES  ${ROOT_LIBRARY_PROPERTIES} )

target_link_libraries(ROOT-8096 ${ARG_LIBRARIES} ${ROOT_LIBRARIES})
set_property(TARGET ROOT-8096 APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME ROOT-8096-build
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                  --target  ROOT-8096${fast} )
#                                  --target  ${targetname_libgen}${fast})

ROOTTEST_ADD_TEST(load
                  MACRO load.C
                  DEPENDS libROOT-8096-build
                  OUTREF load.ref)

ROOTTEST_ADD_TESTDIRS()
