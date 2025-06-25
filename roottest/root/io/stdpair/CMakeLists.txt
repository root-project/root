ROOTTEST_GENERATE_DICTIONARY(libCmsPairCollection
                             cmspair.h
                             LINKDEF FullLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-stdpair-cmspaircoll-fixture)

ROOTTEST_GENERATE_DICTIONARY(libCmsPair
                             cmspair.h LINKDEF PairLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-stdpair-cmspair-fixture)

ROOTTEST_COMPILE_MACRO(cmspair.h
                       FIXTURES_SETUP root-io-stdpair-compile-fixture)

ROOTTEST_ADD_TEST(copyfiles
                  COMMAND ${CMAKE_COMMAND} -E echo "Copying root files"
                  COPY_TO_BUILDDIR aa02.root ab02.root
                  FIXTURES_SETUP root-io-stdpair-inputfiles-fixture)

foreach(A RANGE 0 4)
foreach(B RANGE 0 1)
foreach(C RANGE 0 1)
foreach(D RANGE 0 1)

ROOTTEST_ADD_TEST(pairEnumEvo_${A}_${B}_${C}_${D}
                  MACRO pairEnumEvo.C
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\")"
                  MACROARG ${A},${B},${C},${D}
                  OUTREF pairEnumEvo_${A}_${B}_${C}_${D}.ref
                  FIXTURES_REQUIRED root-io-stdpair-cmspaircoll-fixture
                                    root-io-stdpair-cmspair-fixture
                                    root-io-stdpair-compile-fixture
                                    root-io-stdpair-inputfiles-fixture)

endforeach()
endforeach()
endforeach()
endforeach()
