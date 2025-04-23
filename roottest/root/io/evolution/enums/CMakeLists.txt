# We won't be creating a shared library so we don't want to generate a rootmap file.
set(CMAKE_ROOTTEST_NOROOTMAP ON)
ROOT_GENERATE_DICTIONARY(testVectorOfEnumsDict ${CMAKE_CURRENT_SOURCE_DIR}/classes.hxx
                         LINKDEF classesLinkdef.h)

ROOTTEST_GENERATE_EXECUTABLE(testVectorOfEnums testVectorOfEnums.cxx testVectorOfEnumsDict.cxx
# At least on Windows, we need to explicit list Core, otherwise the linker will fail.
                             LIBRARIES Core RIO Tree
                             FIXTURES_SETUP testVectorOfEnumsExec)

ROOTTEST_ADD_TEST(testWriteVectorOfNonDefaultEnums
                  EXEC ./testVectorOfEnums
                  OPTS w
                  LABELS roottest regression io
                  FIXTURES_SETUP writeNonDefaultEnums
                  FIXTURES_REQUIRED testVectorOfEnumsExec)

ROOTTEST_ADD_TEST(testVectorOfNonDefaultEnums
                  EXEC ./testVectorOfEnums
                  OPTS t
                  COPY_TO_BUILDDIR test_v634.root  treetest_v634.root
                  OUTREF testVectorOfEnums.ref
                  LABELS roottest regression io
                  FIXTURES_REQUIRED writeNonDefaultEnums testVectorOfEnumsExec)

