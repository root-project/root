ROOT_ADD_ROOTTEST(cling-parsing-lessThan MACRO runlessThan.C OUTREF lessThan.ref)
ROOT_ADD_ROOTTEST(cling-parsing-operatorLessThanTemplate MACRO runoperatorLessThanTemplate.C)
ROOT_ADD_ROOTTEST(cling-parsing-testTable MACRO runtestTable.C OUTREF testTable.ref WILLFAIL)

ROOTTEST_GENERATE_DICTIONARY(scopeDict  LINKDEF scopeProblem.C linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(scopeDict2 LINKDEF multiScope.C linkdef.h)

ROOT_ADD_ROOTTEST(cling-parsing-Readbool MACRO runReadbool.C OUTREF Readbool.ref)
ROOT_ADD_ROOTTEST(cling-parsing-ptrconst MACRO ptrconst.C+ OUTREF ptrconst.ref)

ROOTTEST_GENERATE_DICTIONARY(TTestClass_h LINKDEF TTestClass.h)

ROOT_ADD_ROOTTEST(cling-parsing-semicolon MACRO "semicolon.C(\"abc ; xyz\")" OUTREF semicolon.ref)
