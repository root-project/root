#ROOT_COMPILE_MACRO(runvbase.C)

ROOT_ADD_ROOTTEST(cling-offset-vbase-interpreted MACRO runvbase.C OUTREF vbase-i.ref)
ROOT_ADD_ROOTTEST(cling-offset-vbase-compiled MACRO runvbase.C+ OUTREF vbase-c.ref)
