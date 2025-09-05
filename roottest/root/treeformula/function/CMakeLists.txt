ROOTTEST_ADD_TEST(FormulaFuncCall
                  MACRO func.C
                  OUTREF testFormulaFuncCall.ref)

ROOTTEST_COMPILE_MACRO(lista.C
                       FIXTURES_SETUP root-treeformula-function-lista-fixture)

ROOTTEST_ADD_TEST(lista
                  MACRO lista.C+
                  FIXTURES_REQUIRED root-treeformula-function-lista-fixture
                  FIXTURES_SETUP root-treeformula-function-prova-fixture)

ROOTTEST_ADD_TEST(nestedHists
                  MACRO runnestedHists.C
                  OUTREF nestedHists.ref
                  FIXTURES_REQUIRED root-treeformula-function-prova-fixture)

ROOTTEST_ADD_TEST(nestedFunc
                  MACRO runnestedFunc.C
                  OUTREF nestedFunc.ref)

ROOTTEST_COMPILE_MACRO(MyClass.cxx
                       FIXTURES_SETUP root-treeformula-function-MyClass-fixture)

ROOTTEST_ADD_TEST(staticFunc
                  MACRO runstaticFunc.C
                  OUTREF staticFunc.ref
                  FIXTURES_REQUIRED root-treeformula-function-MyClass-fixture)

ROOTTEST_COMPILE_MACRO(all.C
                       FIXTURES_SETUP root-treeformula-function-all-fixture)

ROOTTEST_ADD_TEST(write
                  MACRO runwrite.C
                  FIXTURES_REQUIRED root-treeformula-function-all-fixture
                  FIXTURES_SETUP root-treeformula-function-write-fixture)

ROOTTEST_ADD_TEST(fullload
                  MACRO runfullload.C
                  OUTREF fullload.ref
                  FIXTURES_REQUIRED root-treeformula-function-write-fixture)

ROOTTEST_ADD_TEST(Timestamp
                  MACRO runTimestamp.C
                  OUTREF Timestamp.ref)

ROOTTEST_ADD_TEST(subform
                  MACRO runsubform.C
                  OUTREF subform.ref)

ROOTTEST_ADD_TEST(MinMaxIf
                  MACRO execMinMaxIf.C
                  OUTREF execMinMaxIf.ref)
