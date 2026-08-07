/* @(#)root/multiproc:$Id$ */
// Author: Pere Mato

// Placeholder header file for users accessing TProcPool.h and the class TProcPool

#ifndef __CLING__
//  #pragma message("class 'TProcPool' is deprecated, replace it by 'ROOT::TProcessExecutor' (for TTree processing see 'ROOT:TTreeProcessorMP')")
#endif

#include "ROOT/TProcessExecutor.hxx"

// To keep backward compatibility
using TProcPool = ROOT::TProcessExecutor;

