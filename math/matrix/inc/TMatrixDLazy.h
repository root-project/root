// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDLazy
#define ROOT_TMatrixDLazy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Lazy Matrix classes.                                                 //
//                                                                      //
//  Instantation of                                                     //
//   TMatrixTLazy      <Double_t>                                       //
//   TMatrixTSymLazy   <Double_t>                                       //
//   THaarMatrixT      <Double_t>                                       //
//   THilbertMatrixT   <Double_t>                                       //
//   THilbertMatrixTSym<Double_t>                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixTLazy
#include "TMatrixTLazy.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif
#ifndef ROOT_TMatrixDSymfwd
#include "TMatrixDSymfwd.h"
#endif

typedef TMatrixTLazy      <Double_t> TMatrixDLazy;
typedef TMatrixTSymLazy   <Double_t> TMatrixDSymLazy;
typedef THaarMatrixT      <Double_t> THaarMatrixD;
typedef THilbertMatrixT   <Double_t> THilbertMatrixD;
typedef THilbertMatrixTSym<Double_t> THilbertMatrixDSym;

#endif
