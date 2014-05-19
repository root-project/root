ROOT_ADD_ROOTTEST(root-hist-operator-run MACRO run.C)
ROOT_ADD_ROOTTEST(root-hist-operator-runcopy MACRO runcopy.C+ OUTREF copy.ref)
ROOT_ADD_ROOTTEST(root-hist-operator-runProjections MACRO runProjections.C OUTREF Projections.ref)
