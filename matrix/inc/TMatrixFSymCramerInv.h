// @(#)root/base:$Name:  $:$Id: TMatrixFSymCramerInv.h,v 1.1 2004/10/16 18:09:16 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Oct 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFSymCramerInv
#define ROOT_TMatrixFSymCramerInv

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFSymCramerInv                                                 //
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

#ifndef ROOT_TMatrixFSym
#include "TMatrixFSym.h"
#endif

class TMatrixFSymCramerInv {

public:
  static Bool_t Inv2x2(TMatrixFSym &m,Double_t *determ);
  static Bool_t Inv3x3(TMatrixFSym &m,Double_t *determ);
  static Bool_t Inv4x4(TMatrixFSym &m,Double_t *determ);
  static Bool_t Inv5x5(TMatrixFSym &m,Double_t *determ);
  static Bool_t Inv6x6(TMatrixFSym &m,Double_t *determ);

  virtual ~TMatrixFSymCramerInv() { }
  ClassDef(TMatrixFSymCramerInv,0)  //Cramer Inversion routines for symmetric matrix
};

#endif
