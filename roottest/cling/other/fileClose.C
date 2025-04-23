#include "TROOT.h"
#ifndef __CINT__
static int x = (gROOT->ProcessLine("#define XYZ 21",&x),0);
#endif
