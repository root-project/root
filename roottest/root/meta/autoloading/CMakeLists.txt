ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes
                                    classes.h
                                    SELECTION classes_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-classes-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(ROOT7114classes
                                    ROOT7114classes.h
                                    SELECTION ROOT7114classes_selection.xml
                                    FIXTURES_SETUP root-meta-autoloading-ROOT7114classes-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(TriggerTypeDefs
                                    TriggerTypeDefs.h
                                    SELECTION TriggerTypeDefs_selection.xml
                                    FIXTURES_SETUP root-meta-autoloading-TriggerTypeDefs-fixture)

ROOTTEST_ADD_TEST(ROOT7114
                  MACRO assertROOT7114.C
                  OUTREF assertROOT7114.ref
                  FIXTURES_REQUIRED root-meta-autoloading-ROOT7114classes-fixture)

ROOTTEST_ADD_TEST(AutoLoadEntriesAsSelected
                  MACRO execAutoLoadEntriesAsSelected.C
                  OUTREF execAutoLoadEntriesAsSelected.ref)

ROOTTEST_ADD_TEST(AutoloadOnInclusion
                  MACRO execAutoloadOnInclusion.C
                  COPY_TO_BUILDDIR aHeader.h aRootmap.rootmap
                  OUTREF execAutoloadOnInclusion.ref)

ROOTTEST_ADD_TEST(execTriggerTypedefs
                  MACRO execTriggerTypedefs.C
                  OUTREF execTriggerTypedefs.ref
                  FIXTURES_REQUIRED root-meta-autoloading-TriggerTypeDefs-fixture)

ROOTTEST_ADD_TEST(typeDefAutoloading
                  MACRO exectypeDefAutoloading.C
                  COPY_TO_BUILDDIR exectypeDefAutoloading.rootmap
                  OUTREF exectypeDefAutoloading.ref
                  FIXTURES_REQUIRED root-meta-autoloading-classes-fixture)

ROOTTEST_ADD_TEST(Typeinfo
                  MACRO execTypeinfo.cxx+
                  OUTREF execTypeinfo.ref)

ROOTTEST_ADD_TEST(MissingEntriesROOT-5759
                  MACRO runFullMissingEntriesROOT-5759.C
                  OUTCNVCMD sed -e s,input_line_[0-9]*,input_line_FILTERED,g
                  COPY_TO_BUILDDIR MissingEntriesROOT-5759.rootmap
                  OUTREF MissingEntriesROOT-5759.ref)

#ROOTTEST_ADD_TEST(
#                  MACRO
#                  OUTREF .ref)

#---Copy from source to binary some of the files
#configure_file( . COPYONLY)
#configure_file( . COPYONLY)
#configure_file( . COPYONLY)

#--With ACLiC---------------------------------------------------------
#set(execTypeinfo-suffix +)

#ROOTTEST_ADD_AUTOMACROS(DEPENDS ${depends})
