// @(#)root/hist:$Name:  $:$Id: TUtilHist.cxx,v 1.2 2002/09/15 19:48:47 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// misc histogram utilities                                             //
//                                                                      //
// The functions in this class are called via the TPluginManager.       //
// see TVirtualUtilHist.h for more information .                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TUtilHist.h"
#include "TMatrixFBase.h"
#include "TMatrixDBase.h"
#include "TVectorF.h"
#include "TVectorD.h"
#include "TH2.h"
#include "TF1.h"

ClassImp(TUtilHist)

//______________________________________________________________________________
TUtilHist::TUtilHist() :TVirtualUtilHist()
{
// note that this object is automatically added to the gROOT list of specials
// in the TVirtualUtilHist constructor.
}

//______________________________________________________________________________
TUtilHist::~TUtilHist()
{
}

//______________________________________________________________________________
void TUtilHist::InitStandardFunctions()
{
// to intialize the list of standard functions (poln, gaus, expo, landau)
   
   TF1::InitStandardFunctions();
}

//______________________________________________________________________________
void TUtilHist::PaintMatrix(const TMatrixFBase &m, Option_t *option)
{
// to draw a TMatrixFBase using a TH2F
   
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH2F *R__TMatrixFBase = new TH2F(m);
   R__TMatrixFBase->SetBit(kCanDelete);
   R__TMatrixFBase->Draw(option);
   TH1::AddDirectory(status);   
}

//______________________________________________________________________________
void TUtilHist::PaintMatrix(const TMatrixDBase &m, Option_t *option)
{
// to draw a TMatrixDBase using a TH2D
   
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH2D *R__TMatrixDBase = new TH2D(m);
   R__TMatrixDBase->SetBit(kCanDelete);
   R__TMatrixDBase->Draw(option);
   TH1::AddDirectory(status);   
}

//______________________________________________________________________________
void TUtilHist::PaintVector(const TVectorF &v, Option_t *option)
{
// to draw a TVectorF using a TH1F
   
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH1F *R__TVectorF = new TH1F(v);
   R__TVectorF->SetBit(kCanDelete);
   R__TVectorF->Draw(option);
   TH1::AddDirectory(status);   
}

//______________________________________________________________________________
void TUtilHist::PaintVector(const TVectorD &v, Option_t *option)
{
// to draw a TVectorD using a TH1D
   
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH1D *R__TVectorD = new TH1D(v);
   R__TVectorD->SetBit(kCanDelete);
   R__TVectorD->Draw(option);
   TH1::AddDirectory(status);   
}
