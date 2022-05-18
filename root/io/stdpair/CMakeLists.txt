ROOTTEST_ADD_TESTDIRS()

set(CMAKE_ROOTTEST_NOROOTMAP ON)
set(RootExternalIncludes -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\");")

ROOT_GENERATE_DICTIONARY(G__CmsPairCollection ${CMAKE_CURRENT_SOURCE_DIR}/cmspair.h LINKDEF FullLinkDef.h)
ROOTTEST_LINKER_LIBRARY(CmsPairCollection G__CmsPairCollection.cxx LIBRARIES ${ROOT_LIBRARIES})

ROOT_GENERATE_DICTIONARY(G__CmsPair ${CMAKE_CURRENT_SOURCE_DIR}/cmspair.h LINKDEF PairLinkDef.h)
ROOTTEST_LINKER_LIBRARY(CmsPair G__CmsPair.cxx LIBRARIES ${ROOT_LIBRARIES})

ROOT_ADD_TEST(roottest-root-io-stdpair-collection-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target CmsPairCollection${fast}
              FIXTURES_SETUP InputFiles)

ROOT_ADD_TEST(roottest-root-io-stdpair-pair-build 
              COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target CmsPair${fast}
              FIXTURES_SETUP InputFiles)

ROOTTEST_ADD_TEST(CopyRootFiles
                  COMMAND ${CMAKE_COMMAND} -E echo "Copying root files"
                  COPY_TO_BUILDDIR aa02.root ab02.root
                  FIXTURES_SETUP InputFiles)

ROOTTEST_COMPILE_MACRO(cmspair.h)

foreach(A RANGE 0 4)
foreach(B RANGE 0 1)
foreach(C RANGE 0 1)
foreach(D RANGE 0 1)

ROOTTEST_ADD_TEST(pairEnumEvo_${A}_${B}_${C}_${D}
                  MACRO pairEnumEvo.C
                  MACROARG ${A},${B},${C},${D}
                  OUTREF pairEnumEvo_${A}_${B}_${C}_${D}.ref
                  DEPENDS cmspaircollection-build cmspair-build ${COMPILE_MACRO_TEST}
                  FIXTURES_REQUIRED InputFiles)

endforeach()
endforeach()
endforeach()
endforeach()
