#ROOT_COMPILE_MACRO(rundefaultargs.C)
#ROOT_COMPILE_MACRO(runalgebra.C)

ROOT_ADD_ROOTTEST(cling-dict-defaultargs-interpreted MACRO rundefaultargs.C OUTREF defaultargs_interpreted.ref )
ROOT_ADD_ROOTTEST(cling-dict-defaultargs-compiled MACRO rundefaultargs.C+ OUTREF defaultargs_compiled.ref )
ROOT_ADD_ROOTTEST(cling-dict-templateAutodict MACRO runtemplateAutodict.C OUTREF templateAutodict.ref )
ROOT_ADD_ROOTTEST(cling-dict-operators MACRO runoperators.C OUTREF operators.ref )
ROOT_ADD_ROOTTEST(cling-dict-algebra MACRO runalgebra.C+ OUTREF algebra.ref )
