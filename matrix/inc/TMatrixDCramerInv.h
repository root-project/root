// @(#)root/base:$Name:  $:$Id: TMatrixDCramerInv.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Jan 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDCramerInv
#define ROOT_TMatrixDCramerInv

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDCramerInv                                                    //
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

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

class TMatrixDCramerInv {

public:
  static Bool_t Inv2x2(TMatrixD &m,Double_t *determ);
  static Bool_t Inv3x3(TMatrixD &m,Double_t *determ);
  static Bool_t Inv4x4(TMatrixD &m,Double_t *determ);
  static Bool_t Inv5x5(TMatrixD &m,Double_t *determ);
  static Bool_t Inv6x6(TMatrixD &m,Double_t *determ);

  ClassDef(TMatrixDCramerInv,0)  //Interface to math routines
};

#endif
