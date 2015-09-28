ROOTTEST_ADD_TESTDIRS()

# ROOT-7450
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-privateClassXMLSelection
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/privateClassSelection.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/privateClassSelection_selection.xml
                  ERRREF privateClassXMLSelection.eref)              
                  
# ROOT-7450
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-privateClassFileSelection
                  COPY_TO_BUILDDIR privateClassSelection.h privateClassSelection_LinkDef.h
                  COMMAND ${ROOT_rootcling_CMD} -f out.cpp privateClassSelection.h
                          privateClassSelection_LinkDef.h
                  OUTREF privateClassFileSelection.ref)

# ROOT-7643
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-selectionDuplicate
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/duplicateSelection.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/duplicateSelection_selection.xml
                  ERRREF duplicateSelection.eref
                  PASSRC 1)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-excludedAutoparseKeys
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/excludedAutoparseKeys.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/excludedAutoparseKeys_selection.xml --debug -I ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF  excludedAutoparseKeys.ref
                  OUTCNV  excludedAutoparseKeysOutputFilter.sh)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-unsupportedClasses
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/unsupportedClasses.h
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/unsupportedClasses_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  ERRREF  unsupportedClasses.eref
                  PASSRC 5)

# These 3 tests will fail because a pragma once will be found in the inlined header
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-pragmaOnce1
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce1.h -o  pragmaOnce1.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce1_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-pragmaOnce2
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce2.h -o  pragmaOnce2.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce2_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-pragmaOnce3
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce3.h -o  pragmaOnce3.cxx
                          --select=${CMAKE_CURRENT_SOURCE_DIR}/pragmaOnce3_selection.xml -I ${CMAKE_CURRENT_SOURCE_DIR}
                  WILLFAIL)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(mangledName mangledTest_lhapdf-5.9.1.h  SELECTION mangledTest_lhapdf-5.9.1_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-mangledName
                  MACRO  execmangledTest.C
                  OUTREF execmangledTest.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(MyClass_v1 MyClass_v1.h SELECTION MyClass_v1_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-escapedQuotesRootmapGen
                  MACRO execwriteFirstRootmap.C
                  OUTREF execwriteFirstRootmap.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(MyClass_v2 MyClass_v2.h SELECTION MyClass_v2_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-escapedQuotesWriteTree
                  MACRO execwriteTree.C
                  OUTREF execwriteTree.ref
                  DEPENDS ${GENERATE_REFLEX_TEST} roottest-root-meta-genreflex-escapedQuotesRootmapGen)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-escapedQuotesreadTree
                  MACRO execreadTree.C
                  OUTREF execreadTree.ref
                  DEPENDS roottest-root-meta-genreflex-escapedQuotesWriteTree)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-classSelectedTwice
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/classSelectedTwice.h --verbose --select=${CMAKE_CURRENT_SOURCE_DIR}/classSelectedTwice_selection.xml
                  OUTREF classSelectedTwice.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-virtualInheritance
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/virtualInheritance.h --verbose --select=${CMAKE_CURRENT_SOURCE_DIR}/virtualInheritance_selection.xml
                  OUTREF virtualInheritance.ref)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(typedefSelection typedefSelection.h SELECTION typedefSelection_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-typedefSelection
                  MACRO execTypedefSelection.C
                  OUTREF execTypedefSelection.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(stlPatternSelection stlPatternSelection.h SELECTION stlPatternSelection_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-stlPatternSelection
                  MACRO execstlPatternSelection.C
                  OUTREF execstlPatternSelection.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(complicatedAllocator complicatedAllocator.h SELECTION complicatedAllocator_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-CheckComplicatedAllocator
                  MACRO execCheckComplicatedAllocator.C
                  OUTREF execCheckComplicatedAllocator.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(ns ns.h SELECTION ns_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ns
                  MACRO execns.C
                  OUTREF execns.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(attributesCheck attributesCheck.h SELECTION attributesCheck_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-attributesCheck
                  MACRO execAttributesCheck.C
                  OUTREF execAttributesCheck.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
