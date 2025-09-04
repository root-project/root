#include "TROOT.h"
#ifndef __CLING__
static int x = (gROOT->ProcessLine("#define XYZ 21",&x),0);
#endif
