#TEST_TARGETS += defaultargs templateAutodict operators algebra

ROOT_ADD_ROOTTEST(cint-dict-defaultargs-interpreted MACRO rundefaultargs.C OUTREF defaultargs_interpreted.ref )
ROOT_ADD_ROOTTEST(cint-dict-defaultargs-compiled MACRO rundefaultargs.C+ OUTREF defaultargs_compiled.ref )
ROOT_ADD_ROOTTEST(cint-dict-templateAutodict MACRO runtemplateAutodict.C OUTREF templateAutodict.ref )
ROOT_ADD_ROOTTEST(cint-dict-operators MACRO runoperators.C OUTREF operators.ref )
ROOT_ADD_ROOTTEST(cint-dict-algebra MACRO runalgebra.C+ OUTREF algebra.ref )

