ROOT_GENERATE_DICTIONARY(G__TreeFormulaRetobjGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/EventLinkDef.h)

ROOT_LINKER_LIBRARY(TreeFormulaRetobjGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/TreeFormulaRetobjGeneration.cxx
  G__TreeFormulaRetobjGeneration.cxx
  LIBRARIES Core Tree Hist MathCore)

if(MSVC)
  add_custom_command(TARGET TreeFormulaRetobjGeneration POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libTreeFormulaRetobjGeneration.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libTreeFormulaRetobjGeneration.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libTreeFormulaRetobjGeneration.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/libTreeFormulaRetobjGeneration.lib)
endif()

ROOTTEST_GENERATE_EXECUTABLE(TreeFormulaRetobjGenerator
  ${CMAKE_CURRENT_SOURCE_DIR}/TreeFormulaRetobjGenerator.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore TreeFormulaRetobjGeneration
  DEPENDS TreeFormulaRetobjGeneration
  FIXTURES_SETUP treeformula-retobj-generator)

ROOTTEST_ADD_TEST(retobj
  COPY_TO_BUILDDIR helper.C mcpool.root
  PRECMD ./TreeFormulaRetobjGenerator 10 1 0 1
  MACRO   runretobjTest.C
  OUTREF  retobjTest.ref
  FIXTURES_REQUIRED treeformula-retobj-generator)

ROOTTEST_ADD_TEST(enum
  MACRO  runenum.C
  OUTREF enum.ref)
