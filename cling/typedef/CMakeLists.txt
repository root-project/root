ROOT_COMPILE_MACRO(vclass.h) 
ROOT_ADD_ROOTTEST(cling-typedef-ptrptr_test MACRO runptrptr.C OUTREF ptrptr.ref OUTCNVCMD grep ptrptr.C)
