# ROOT-8096

add_custom_command(OUTPUT dict.cxx
   COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/definitions.h -o dict.cxx  --selection_file=${CMAKE_CURRENT_SOURCE_DIR}/selection.xml --rootmap libROOT-8096.rootmap --rootmap-lib libROOT-8096${libsuffix} )

set_source_files_properties(dict.cxx PROPERTIES GENERATED TRUE)

add_library(ROOT-8096 EXCLUDE_FROM_ALL SHARED dict.cxx)
set_target_properties(ROOT-8096 PROPERTIES  ${ROOT_LIBRARY_PROPERTIES} )

target_link_libraries(ROOT-8096 ${ARG_LIBRARIES} ${ROOT_LIBRARIES})
set_property(TARGET ROOT-8096 APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES dict.cxx dict_rdict.pcm libROOT-8096.rootmap)

add_test(NAME roottest-cling-dict-ROOT-8096-build
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                  --config $<CONFIG>
                                  --target  ROOT-8096${fast}
                                  -- ${always-make})
#                                  --target  ${targetname_libgen}${fast})
set_property(TEST roottest-cling-dict-ROOT-8096-build PROPERTY ENVIRONMENT ${ROOTTEST_ENVIRONMENT})

if(MSVC)
   set(load_precmd ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libROOT-8096.dll ${CMAKE_CURRENT_BINARY_DIR}/)
endif()
ROOTTEST_ADD_TEST(load
                  MACRO load.C
                  PRECMD ${load_precmd}
                  DEPENDS roottest-cling-dict-ROOT-8096-build
                  OUTREF load.ref)

ROOTTEST_ADD_TESTDIRS()
