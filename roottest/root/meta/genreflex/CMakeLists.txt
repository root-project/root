ROOTTEST_ADD_TESTDIRS()

# ROOT-9872
ROOTTEST_GENERATE_REFLEX_DICTIONARY(ROOT_9872 classes_ROOT_9872.h SELECTION classes_def_ROOT_9872.xml)

ROOTTEST_ADD_TEST(ROOT_9872
                  MACRO run_ROOT_9872.C
                  DEPENDS ${GENERATE_REFLEX_TEST})

# ROOT-8509
ROOTTEST_ADD_TEST(noIncludePaths
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/noIncludePaths.h
                          --noIncludePaths --select=${CMAKE_CURRENT_SOURCE_DIR}/noIncludePaths_selection.xml
                  )

# ROOT-8380
ROOTTEST_GENERATE_REFLEX_DICTIONARY(canSplit canSplit.h SELECTION canSplit_selection.xml)

ROOTTEST_ADD_TEST(canSplit
                  MACRO canSplit.C
                  DEPENDS ${GENERATE_REFLEX_TEST})


# ROOT-8381
ROOTTEST_ADD_TEST(vectorOfTStar
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/vectorOfTStar.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/vectorOfTStar_selection.xml
                  OUTREF vectorOfTStar.ref
                  ERRREF vectorOfTStar.eref)

# ROOT-3745
ROOTTEST_ADD_TEST(ioreadrulesfailonwarning
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/ioreadrulesfailonwarning.h
                          --fail_on_warnings --select=${CMAKE_CURRENT_SOURCE_DIR}/ioreadrulesfailonwarning_selection.xml
                  OUTREF ioreadrulesfailonwarning.eref
                  PASSRC 1)

#ROOT-7760
# This test runs only on apple platforms as it probes a platform specific feature.
if(APPLE)
  ROOTTEST_ADD_TEST(checkOSXLibraryExtension
                    COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/checkOSXLibraryExtension.h
                            -l libMylib.dylib --rootmap checkOSXLibraryExtension.rootmap --rootmap-lib libMylib.dylib
                    OUTREF checkOSXLibraryExtension.ref)
endif()

ROOTTEST_ADD_TEST(conflictingSelections
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/conflictingNameRule.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/conflictingNameRule_selection.xml
                  OUTREF conflictingNameRule.eref
                  ${WILLFAIL_ON_WIN32}
                  PASSRC 1)

ROOTTEST_ADD_TEST(equivalentSelections
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/equivalentSelections.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/equivalentSelections_selection.xml
                  ${WILLFAIL_ON_WIN32}
                  OUTREF equivalentSelections.eref)

#ROOT-7676
ROOTTEST_ADD_TEST(twoTypedefsOneClass
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/twoTypedefsOneClass.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/twoTypedefsOneClass_selection.xml
                  OUTREF twoTypedefsOneClass.ref)

# ROOT-7450
ROOTTEST_ADD_TEST(privateClassXMLSelection
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/privateClassSelection.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/privateClassSelection_selection.xml
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
                  ERRREF duplicateSelection.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(excludedAutoparseKeys
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/excludedAutoparseKeys.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/excludedAutoparseKeys_selection.xml --debug -I ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF  excludedAutoparseKeys.ref
                  OUTCNV  excludedAutoparseKeysOutputFilter.sh)

ROOTTEST_ADD_TEST(unsupportedClasses
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/unsupportedClasses.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/unsupportedClasses_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  ERRREF  unsupportedClasses.eref
                  PASSRC 4)

# These 3 tests will fail because a pragma once will be found in the inlined header
ROOTTEST_ADD_TEST(pragmaOnce1
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce1.h -o  pragmaOnce1.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce1_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_ADD_TEST(pragmaOnce2
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce2.h -o  pragmaOnce2.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce2_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_ADD_TEST(pragmaOnce3
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce3.h -o  pragmaOnce3.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce3_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(mangledName mangledTest_lhapdf-5.9.1.h  SELECTION mangledTest_lhapdf-5.9.1_selection.xml)

ROOTTEST_ADD_TEST(mangledName
                  MACRO  execmangledTest.C
                  OUTREF execmangledTest.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(MyClass_v1 MyClass_v1.h SELECTION MyClass_v1_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(escapedQuotesRootmapGen
                  MACRO execwriteFirstRootmap.C
                  OUTREF execwriteFirstRootmap.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(MyClass_v2 MyClass_v2.h SELECTION MyClass_v2_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(escapedQuotesWriteTree
                  MACRO execwriteTree.C
                  OUTREF execwriteTree.ref
                  DEPENDS ${GENERATE_REFLEX_TEST} escapedQuotesRootmapGen)

ROOTTEST_ADD_TEST(escapedQuotesreadTree
                  MACRO execreadTree.C
                  OUTREF execreadTree.ref
                  DEPENDS escapedQuotesWriteTree)

ROOTTEST_ADD_TEST(classSelectedTwice
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classSelectedTwice.h --verbose --select=${CMAKE_CURRENT_SOURCE_DIR}/classSelectedTwice_selection.xml
                  ${WILLFAIL_ON_WIN32}
                  OUTREF classSelectedTwice.ref)

ROOTTEST_ADD_TEST(virtualInheritance
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/virtualInheritance.h --verbose --select=${CMAKE_CURRENT_SOURCE_DIR}/virtualInheritance_selection.xml
                  OUTREF virtualInheritance.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(typedefSelection typedefSelection.h SELECTION typedefSelection_selection.xml)

ROOTTEST_ADD_TEST(typedefSelection
                  MACRO execTypedefSelection.C
                  OUTREF execTypedefSelection.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(stlPatternSelection stlPatternSelection.h SELECTION stlPatternSelection_selection.xml)

ROOTTEST_ADD_TEST(stlPatternSelection
                  MACRO execstlPatternSelection.C
                  OUTREF execstlPatternSelection.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(complicatedAllocator complicatedAllocator.h SELECTION complicatedAllocator_selection.xml)

ROOTTEST_ADD_TEST(CheckComplicatedAllocator
                  MACRO execCheckComplicatedAllocator.C
                  OUTREF execCheckComplicatedAllocator.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(ns ns.h SELECTION ns_selection.xml)

ROOTTEST_ADD_TEST(ns
                  MACRO execns.C
                  OUTREF execns.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(attributesCheck attributesCheck.h SELECTION attributesCheck_selection.xml)

ROOTTEST_ADD_TEST(attributesCheck
                  MACRO execAttributesCheck.C
                  OUTREF execAttributesCheck.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
