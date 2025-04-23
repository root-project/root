ROOTTEST_ADD_TEST(old-formula
  PRECMD ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/hsimple.31000.root
    ${CMAKE_CURRENT_BINARY_DIR}/hsimple.31000.root
  MACRO runoldFormulaTest.C
  OUTREF oldFormulaTest.ref)
