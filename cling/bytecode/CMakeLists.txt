ROOT_ADD_ROOTTEST(cling-bytecode-complex MACRO  runcomplex.C OUTREF complex.ref)
ROOT_ADD_ROOTTEST(cling-bytecode-henry MACRO  runhenry.C OUTREF henry.ref)
ROOT_ADD_ROOTTEST(cling-bytecode-arrayhist MACRO  runarrayhist.C OUTREF arrayhist.ref WILLFAIL)
