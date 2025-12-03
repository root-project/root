# We need to be able to specify the name of the library fully, no
# suffixes/prefixes automatically added, if we want compatibility
# with roottest. Indeed roottest does not allow
# dynamic rootmap creation: you have to commit one in the repository. If ctest
# creates another rootmap with the same keys but a different library name the
# system will complain at startup, making the test fail. Clearly for backwards
# compatibility reasons the current behaviour should be preserved :)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(FullheaderParsingOnDemand
                                    FullheaderParsingOnDemand.h
                                    SELECTION FullheaderParsingOnDemand_selection.xml
                                    LIBNAME libFullheaderParsingOnDemand_dictrflx
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-headerParsingOnDemand-dic1-fixture)

ROOTTEST_ADD_TEST(runFullheaderParsingOnDemand
                  COPY_TO_BUILDDIR headerParsingOnDemand.rootmap
                  MACRO runFullheaderParsingOnDemand.C
                  OUTREF headerParsingOnDemand.ref
                  OUTCNV FullheaderParsingOnDemand_convert.sh
                  FIXTURES_REQUIRED root-meta-autoloading-headerParsingOnDemand-dic1-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(complexTypedefs
                                    complexTypedefs.h
                                    SELECTION complexTypedefs_selection.xml
                                    LIBNAME libcomplexTypedefs_dictrflx
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-headerParsingOnDemand-dic2-fixture)

ROOTTEST_ADD_TEST(execCheckComplexTypedefs
                  COPY_TO_BUILDDIR complexTypedefs.rootmap
                  MACRO execCheckComplexTypedefs.C
                  OUTREF execCheckComplexTypedefs.ref
                  ${WILLFAIL_ON_WIN32}
                  FIXTURES_REQUIRED root-meta-autoloading-headerParsingOnDemand-dic2-fixture)

ROOTTEST_ADD_TEST(execLDAPAttribute
                  MACRO execLDAPAttribute.C
                  OUTREF execLDAPAttribute.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(scopedEnums
                                    scopedEnums.h
                                    SELECTION scopedEnums_selection.xml
                                    LIBNAME libscopedEnums_dictrflx
                                    FIXTURES_SETUP root-meta-autoloading-headerParsingOnDemand-dic3-fixture)

ROOTTEST_ADD_TEST(runscopedEnums
                  MACRO runscopedEnums.C
                  OUTREF scopedEnums.ref
                  OUTCNV scopedEnums_convert.sh
                  FIXTURES_REQUIRED root-meta-autoloading-headerParsingOnDemand-dic3-fixture)

ROOTTEST_GENERATE_DICTIONARY(libno_autoparse_v10
                             no_autoparse.h
                             OPTIONS -DITEM_V10
                             SOURCES no_autoparse_writer.cxx
                             LINKDEF no_autoparseLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-autoloading-headerParsingOnDemand-dicv10-fixture)

target_compile_definitions(libno_autoparse_v10libgen PUBLIC ITEM_V10)

ROOTTEST_GENERATE_DICTIONARY(libno_autoparse_v11
                             no_autoparse.h
                             OPTIONS -DITEM_V11
                             SOURCES no_autoparse_writer.cxx
                             LINKDEF no_autoparseLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-autoloading-headerParsingOnDemand-dicv11-fixture)

target_compile_definitions(libno_autoparse_v11libgen PUBLIC ITEM_V11)

ROOTTEST_ADD_TEST(no_autoparse_write
                  MACRO no_autoparse_write.C
                  FIXTURES_REQUIRED root-meta-autoloading-headerParsingOnDemand-dicv10-fixture
                  FIXTURES_SETUP root-meta-autoloading-headerParsingOnDemand-no_autoparse_write-fixture)

ROOTTEST_ADD_TEST(no_autoparse_read
                  MACRO no_autoparse_read.cxx+
                  FIXTURES_REQUIRED root-meta-autoloading-headerParsingOnDemand-dicv11-fixture
                                    root-meta-autoloading-headerParsingOnDemand-no_autoparse_write-fixture)
