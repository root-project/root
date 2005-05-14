// @(#)root/base:$Name:  $:$Id: TMatrixDSymCramerInv.h,v 1.1 2004/10/16 18:09:16 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Oct 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDSymCramerInv
#define ROOT_TMatrixDSymCramerInv

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSymCramerInv                                                 //
//                                                                      //
// Encapsulate Cramer Inversion routines.                               //
//                                                                      //
// The 4x4, 5x5 and 6x6 are adapted from routines written by            //
// Mark Fischler and Steven Haywood as part of the CLHEP package        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TMatrixDSym
#include "TMatrixDSym.h"
#endif

class TMatrixDSymCramerInv {

public:
  static Bool_t Inv2x2(TMatrixDSym &m,Double_t *determ);
  static Bool_t Inv3x3(TMatrixDSym &m,Double_t *determ);
  static Bool_t Inv4x4(TMatrixDSym &m,Double_t *determ);
  static Bool_t Inv5x5(TMatrixDSym &m,Double_t *determ);
  static Bool_t Inv6x6(TMatrixDSym &m,Double_t *determ);

  virtual ~TMatrixDSymCramerInv() { }
  ClassDef(TMatrixDSymCramerInv,0)  //Cramer Inversion routines for symmetric matrix
};

#endif
