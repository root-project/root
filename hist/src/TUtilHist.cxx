// @(#)root/hist:$Name:  $:$Id: TUtilHist.cxx,v 1.1 2002/09/14 16:19:14 brun Exp $
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
//////////////////////////////////////////////////////////////////////////

#include "TUtilHist.h"
#include "TROOT.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TVector.h"
#include "TVectorD.h"
#include "TH2.h"
#include "TF1.h"

ClassImp(TUtilHist)

//______________________________________________________________________________
TUtilHist::TUtilHist()
{
   SetName("R__TVirtualUtilHist");
   gROOT->GetListOfSpecials()->Add(this);
}

//______________________________________________________________________________
TUtilHist::~TUtilHist()
{
}

//______________________________________________________________________________
void TUtilHist::InitStandardFunctions()
{
   TF1::InitStandardFunctions();
}

//______________________________________________________________________________
void TUtilHist::PaintMatrix(TMatrix &m, Option_t *option)
{
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH2F *R__TMatrix = new TH2F(m);
   R__TMatrix->SetBit(kCanDelete);
   R__TMatrix->Draw(option);
   TH1::AddDirectory(status);   
}

//______________________________________________________________________________
void TUtilHist::PaintMatrix(TMatrixD &m, Option_t *option)
{
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH2D *R__TMatrixD = new TH2D(m);
   R__TMatrixD->SetBit(kCanDelete);
   R__TMatrixD->Draw(option);
   TH1::AddDirectory(status);   
}

//______________________________________________________________________________
void TUtilHist::PaintVector(TVector &v, Option_t *option)
{
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH1F *R__TVector = new TH1F(v);
   R__TVector->SetBit(kCanDelete);
   R__TVector->Draw(option);
   TH1::AddDirectory(status);   
}

//______________________________________________________________________________
void TUtilHist::PaintVector(TVectorD &v, Option_t *option)
{
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH1D *R__TVectorD = new TH1D(v);
   R__TVectorD->SetBit(kCanDelete);
   R__TVectorD->Draw(option);
   TH1::AddDirectory(status);   
}
