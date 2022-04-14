ROOTTEST_ADD_TESTDIRS()

set(CMAKE_ROOTTEST_NOROOTMAP ON)
set(RootExternalIncludes -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\");")

ROOT_GENERATE_DICTIONARY(G__CmsPairCollection ${CMAKE_CURRENT_SOURCE_DIR}/cmspair.h LINKDEF FullLinkDef.h)
ROOT_LINKER_LIBRARY(CmsPairCollection TEST G__CmsPairCollection.cxx LIBRARIES ${ROOT_LIBRARIES})

ROOT_GENERATE_DICTIONARY(G__CmsPair ${CMAKE_CURRENT_SOURCE_DIR}/cmspair.h LINKDEF PairLinkDef.h)
ROOT_LINKER_LIBRARY(CmsPair TEST G__CmsPair.cxx LIBRARIES ${ROOT_LIBRARIES})

ROOT_ADD_TEST(roottest-root-io-stdpair-collection-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target CmsPairCollection${fast})

ROOT_ADD_TEST(roottest-root-io-stdpair-pair-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target CmsPair${fast})

ROOTTEST_COMPILE_MACRO(cmspair.h)

foreach(A RANGE 0 3)
foreach(B RANGE 0 1)
foreach(C RANGE 0 1)
foreach(D RANGE 0 1)

ROOTTEST_ADD_TEST(pairEnumEvo_${A}_${B}_${C}_${D}
                  MACRO pairEnumEvo.C
                  MACROARG ${A},${B},${C},${D}
                  COPY_TO_BUILDDIR aa02.root ab02.root
                  OUTREF pairEnumEvo_${A}_${B}_${C}_${D}.ref
                  DEPENDS cmspaircollection-build cmspair-build ${COMPILE_MACRO_TEST})

endforeach()
endforeach()
endforeach()
endforeach()
