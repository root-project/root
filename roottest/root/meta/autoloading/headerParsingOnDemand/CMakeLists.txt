# We need to be able to specify the name of the library fully, no
# suffixes/prefixes automatically added, if we want compatibility
# with roottest. Indeed roottest does not allow
# dynamic rootmap creation: you have to commit one in the repository. If ctest
# creates another rootmap with the same keys but a different library name the
# system will complain at startup, making the test fail. Clearly for backwards
# compatibility reasons the current behaviour should be preserved :)

# disable tests which are failing on Windows because of their PATH being too long
# (>260 characters) when run in the CI (Jenkins)
if(NOT MSVC OR NOT DEFINED ENV{JENKINS_HOME} OR win_broken_tests)
  ROOTTEST_GENERATE_REFLEX_DICTIONARY(FullheaderParsingOnDemand
                                      FullheaderParsingOnDemand.h
                                      SELECTION FullheaderParsingOnDemand_selection.xml
                                      LIBNAME libFullheaderParsingOnDemand_dictrflx
                                      NO_ROOTMAP)

  ROOTTEST_ADD_TEST(runFullheaderParsingOnDemand
                    COPY_TO_BUILDDIR headerParsingOnDemand.rootmap
                    MACRO runFullheaderParsingOnDemand.C
                    OUTREF headerParsingOnDemand.ref
                    OUTCNV FullheaderParsingOnDemand_convert.sh
                    DEPENDS ${GENERATE_REFLEX_TEST})
endif()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(complexTypedefs
                                    complexTypedefs.h
                                    SELECTION complexTypedefs_selection.xml
                                    LIBNAME libcomplexTypedefs_dictrflx
                                    NO_ROOTMAP)

ROOTTEST_ADD_TEST(execCheckComplexTypedefs
                  COPY_TO_BUILDDIR complexTypedefs.rootmap
                  MACRO execCheckComplexTypedefs.C
                  OUTREF execCheckComplexTypedefs.ref
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(execLDAPAttribute
                  MACRO execLDAPAttribute.C
                  OUTREF execLDAPAttribute.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(scopedEnums
                                    scopedEnums.h
                                    SELECTION scopedEnums_selection.xml
                                    LIBNAME libscopedEnums_dictrflx)

ROOTTEST_ADD_TEST(runscopedEnums
                  MACRO runscopedEnums.C
                  OUTREF scopedEnums.ref
                  OUTCNV scopedEnums_convert.sh
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_DICTIONARY(libno_autoparse_v10
                             no_autoparse.h
                             OPTIONS -DITEM_V10
                             LINKDEF no_autoparseLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE)

target_compile_definitions(libno_autoparse_v10libgen PUBLIC ITEM_V10)
target_sources(libno_autoparse_v10libgen PRIVATE no_autoparse_writer.cxx)

ROOTTEST_GENERATE_DICTIONARY(libno_autoparse_v11
                             no_autoparse.h
                             OPTIONS -DITEM_V11
                             LINKDEF no_autoparseLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE)

target_compile_definitions(libno_autoparse_v11libgen PUBLIC ITEM_V11)
target_sources(libno_autoparse_v11libgen PRIVATE no_autoparse_writer.cxx)

ROOTTEST_ADD_TEST(no_autoparse_write
                  MACRO no_autoparse_write.C
                  DEPENDS libno_autoparse_v10-build)

ROOTTEST_ADD_TEST(no_autoparse_read
                  MACRO no_autoparse_read.cxx+
                  DEPENDS libno_autoparse_v11-build no_autoparse_write)
