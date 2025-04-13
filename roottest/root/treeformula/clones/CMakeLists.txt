ROOTTEST_ADD_TEST(nested-2
  PRECMD ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/digi.root
    ${CMAKE_CURRENT_BINARY_DIR}/digi.root
  MACRO digiRun.C
  OUTREF nestedClones2.ref)

ROOTTEST_ADD_TEST(nested-3
  PRECMD ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/nesting3.root
    ${CMAKE_CURRENT_BINARY_DIR}/nesting3.root
  MACRO runNestedClones3.C
  OUTREF nestedClones3.ref)
