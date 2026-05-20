# List only header.h which transitively includes enum.h - this leads to two
# __attribute__((annotate("$clingAutoload$...")))
ROOTTEST_GENERATE_DICTIONARY(enumDict header.h LINKDEF LinkDef.h NO_CXXMODULE)

ROOTTEST_ADD_TEST(exec
    MACRO execEnumDict.C
    DEPENDS ${GENERATE_DICTIONARY_TEST}
)
