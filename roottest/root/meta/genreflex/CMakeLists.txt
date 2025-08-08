ROOTTEST_ADD_TESTDIRS()

# ROOT-9872
ROOTTEST_GENERATE_REFLEX_DICTIONARY(ROOT_9872
                                    classes_ROOT_9872.h
                                    SELECTION classes_def_ROOT_9872.xml
                                    FIXTURES_SETUP root-meta-genreflex-ROOT_9872-fixture)

ROOTTEST_ADD_TEST(ROOT_9872
                  MACRO run_ROOT_9872.C
                  FIXTURES_REQUIRED root-meta-genreflex-ROOT_9872-fixture)

# ROOT-8509
ROOTTEST_ADD_TEST(noIncludePaths
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/noIncludePaths.h
                          --noIncludePaths --select=${CMAKE_CURRENT_SOURCE_DIR}/noIncludePaths_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/noIncludePaths_rflx.cxx
                  )

# ROOT-8380
ROOTTEST_GENERATE_REFLEX_DICTIONARY(canSplit
                                    canSplit.h
                                    SELECTION canSplit_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-canSplit-fixture)

ROOTTEST_ADD_TEST(canSplit
                  MACRO canSplit.C
                  FIXTURES_REQUIRED root-meta-genreflex-canSplit-fixture)


# ROOT-8381
ROOTTEST_ADD_TEST(vectorOfTStar
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/vectorOfTStar.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/vectorOfTStar_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/vectorOfTStar_rflx.cxx
                  OUTREF vectorOfTStar.ref
                  ERRREF vectorOfTStar.eref)

# ROOT-3745
ROOTTEST_ADD_TEST(ioreadrulesfailonwarning
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/ioreadrulesfailonwarning.h
                          --fail_on_warnings --select=${CMAKE_CURRENT_SOURCE_DIR}/ioreadrulesfailonwarning_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/ioreadrulesfailonwarning_rflx.cxx
                  OUTREF ioreadrulesfailonwarning.eref
                  PASSRC 1)

#ROOT-7760
# This test runs only on apple platforms as it probes a platform specific feature.
if(APPLE)
  ROOTTEST_ADD_TEST(checkOSXLibraryExtension
                    COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/checkOSXLibraryExtension.h
                            -l libMylib.dylib --rootmap checkOSXLibraryExtension.rootmap --rootmap-lib libMylib.dylib
                            --output=${CMAKE_CURRENT_BINARY_DIR}/checkOSXLibraryExtension_rflx.cxx
                    OUTREF checkOSXLibraryExtension.ref)
endif()

ROOTTEST_ADD_TEST(conflictingSelections
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/conflictingNameRule.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/conflictingNameRule_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/conflictingNameRule_rflx.cxx
                  OUTREF conflictingNameRule.eref
                  ${WILLFAIL_ON_WIN32}
                  PASSRC 1)

ROOTTEST_ADD_TEST(equivalentSelections
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/equivalentSelections.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/equivalentSelections_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/equivalentSelections_rflx.cxx
                  ${WILLFAIL_ON_WIN32}
                  OUTREF equivalentSelections.eref)

#ROOT-7676
ROOTTEST_ADD_TEST(twoTypedefsOneClass
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/twoTypedefsOneClass.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/twoTypedefsOneClass_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/twoTypedefsOneClass_rflx.cxx
                  OUTREF twoTypedefsOneClass.ref)

# ROOT-7450
ROOTTEST_ADD_TEST(privateClassXMLSelection
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/privateClassSelection.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/privateClassSelection_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/privateClassSelection_rflx.cxx
                  ERRREF privateClassXMLSelection.eref)

# ROOT-7450
ROOTTEST_ADD_TEST(privateClassFileSelection
                  COPY_TO_BUILDDIR privateClassSelection.h privateClassSelection_LinkDef.h
                  COMMAND ${ROOT_rootcling_CMD} -f out.cpp privateClassSelection.h
                          privateClassSelection_LinkDef.h
                  OUTREF privateClassFileSelection.ref)

# ROOT-7643
ROOTTEST_ADD_TEST(selectionDuplicate
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/duplicateSelection.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/duplicateSelection_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/duplicateSelection_rflx.cxx
                  ERRREF duplicateSelection.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(excludedAutoparseKeys
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/excludedAutoparseKeys.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/excludedAutoparseKeys_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/excludedAutoparseKeys_rflx.cxx
                          --debug -I ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF  excludedAutoparseKeys.ref
                  OUTCNV  excludedAutoparseKeysOutputFilter.sh)

ROOTTEST_ADD_TEST(unsupportedClasses
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/unsupportedClasses.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/unsupportedClasses_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/unsupportedClasses_rflx.cxx
                          -I ${CMAKE_CURRENT_SOURCE_DIR}
                  ERRREF  unsupportedClasses.eref
                  PASSRC 4)

# These 3 tests will fail because a pragma once will be found in the inlined header
ROOTTEST_ADD_TEST(pragmaOnce1
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce1.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce1_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/pragmaOnce1.cxx
                          -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_ADD_TEST(pragmaOnce2
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce2.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce2_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/pragmaOnce2.cxx
                          -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_ADD_TEST(pragmaOnce3
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce3.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce3_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/pragmaOnce3.cxx
                          -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(mangledName
                                    mangledTest_lhapdf-5.9.1.h
                                    SELECTION mangledTest_lhapdf-5.9.1_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-mangledName-fixture)

ROOTTEST_ADD_TEST(mangledName
                  MACRO  execmangledTest.C
                  OUTREF execmangledTest.ref
                  FIXTURES_REQUIRED root-meta-genreflex-mangledName-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(MyClass_v1
                                    MyClass_v1.h
                                    SELECTION MyClass_v1_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-genreflex-MyClass_v1-fixture)

ROOTTEST_ADD_TEST(escapedQuotesRootmapGen
                  MACRO execwriteFirstRootmap.C
                  OUTREF execwriteFirstRootmap.ref
                  FIXTURES_REQUIRED root-meta-genreflex-MyClass_v1-fixture
                  FIXTURES_SETUP root-meta-genreflex-escapedQuotesRootmapGen-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(MyClass_v2
                                    MyClass_v2.h
                                    SELECTION MyClass_v2_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-genreflex-MyClass_v2-fixture)

ROOTTEST_ADD_TEST(escapedQuotesWriteTree
                  MACRO execwriteTree.C
                  OUTREF execwriteTree.ref
                  FIXTURES_REQUIRED root-meta-genreflex-escapedQuotesRootmapGen-fixture
                                    root-meta-genreflex-MyClass_v2-fixture
                  FIXTURES_SETUP root-meta-genreflex-escapedQuotesWriteTree-fixture)

ROOTTEST_ADD_TEST(escapedQuotesreadTree
                  MACRO execreadTree.C
                  OUTREF execreadTree.ref
                  FIXTURES_REQUIRED root-meta-genreflex-escapedQuotesWriteTree-fixture)

ROOTTEST_ADD_TEST(classSelectedTwice
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classSelectedTwice.h --verbose
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/classSelectedTwice_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/classSelectedTwice_rflx.cxx
                  ${WILLFAIL_ON_WIN32}
                  OUTREF classSelectedTwice.ref)

ROOTTEST_ADD_TEST(virtualInheritance
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/virtualInheritance.h --verbose
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/virtualInheritance_selection.xml
                          --output=${CMAKE_CURRENT_BINARY_DIR}/virtualInheritance_rflx.cxx
                  OUTREF virtualInheritance.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(typedefSelection
                                    typedefSelection.h
                                    SELECTION typedefSelection_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-typedefSelection-fixture)

ROOTTEST_ADD_TEST(typedefSelection
                  MACRO execTypedefSelection.C
                  OUTREF execTypedefSelection.ref
                  FIXTURES_REQUIRED root-meta-genreflex-typedefSelection-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(stlPatternSelection
                                    stlPatternSelection.h
                                    SELECTION stlPatternSelection_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-stlPatternSelection-fixture)

ROOTTEST_ADD_TEST(stlPatternSelection
                  MACRO execstlPatternSelection.C
                  OUTREF execstlPatternSelection.ref
                  FIXTURES_REQUIRED root-meta-genreflex-stlPatternSelection-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(complicatedAllocator
                                    complicatedAllocator.h
                                    SELECTION complicatedAllocator_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-complicatedAllocator-fixture)

ROOTTEST_ADD_TEST(CheckComplicatedAllocator
                  MACRO execCheckComplicatedAllocator.C
                  OUTREF execCheckComplicatedAllocator.ref
                  FIXTURES_REQUIRED root-meta-genreflex-complicatedAllocator-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(ns
                                    ns.h
                                    SELECTION ns_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-ns-fixture)

ROOTTEST_ADD_TEST(ns
                  MACRO execns.C
                  OUTREF execns.ref
                  FIXTURES_REQUIRED root-meta-genreflex-ns-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(attributesCheck
                                    attributesCheck.h
                                    SELECTION attributesCheck_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-attributesCheck-fixture)

ROOTTEST_ADD_TEST(attributesCheck
                  MACRO execAttributesCheck.C
                  OUTREF execAttributesCheck.ref
                  FIXTURES_REQUIRED root-meta-genreflex-attributesCheck-fixture)
