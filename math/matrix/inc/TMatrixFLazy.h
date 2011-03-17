// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFLazy
#define ROOT_TMatrixFLazy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Lazy Matrix classes.                                                 //
//                                                                      //
//  Instantation of                                                     //
//   TMatrixTLazy      <Float_t>                                        //
//   TMatrixTSymLazy   <Float_t>                                        //
//   THaarMatrixT      <Float_t>                                        //
//   THilbertMatrixT   <Float_t>                                        //
//   THilbertMatrixTSym<Float_t>                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixTLazy
#include "TMatrixTLazy.h"
#endif
#ifndef ROOT_TMatrixFfwd
#include "TMatrixFfwd.h"
#endif
#ifndef ROOT_TMatrixFSymfwd
#include "TMatrixFSymfwd.h"
#endif

typedef TMatrixTLazy      <Float_t> TMatrixFLazy;
typedef TMatrixTSymLazy   <Float_t> TMatrixFSymLazy;
typedef THaarMatrixT      <Float_t> THaarMatrixF;
typedef THilbertMatrixT   <Float_t> THilbertMatrixF;
typedef THilbertMatrixTSym<Float_t> THilbertMatrixFSym;

#endif
