/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.cc,v 1.1 2001/03/28 19:21:48 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

#include "RooFitCore/RooHist.hh"
//#include "RooFitTools/RooMath.hh"

ClassImp(RooHist)

static const char rcsid[] =
"$Id: RooHist.cc,v 1.1 2001/03/28 19:21:48 davidk Exp $";

RooHist::RooHist(Double_t nSigma) :
  TGraphAsymmErrors(), _nSigma(nSigma)
{
  initialize();
}

void RooHist::initialize() {
  _ymax= 0;
  SetMarkerStyle(8);
  SetDrawOption("P");
}

void RooHist::addBin(Float_t binCenter, Int_t n) {
  Int_t index= GetN();
//    Double_t ym= RooMath::PoissonError(n,RooMath::NegativeError,_nSigma);
//    Double_t yp= RooMath::PoissonError(n,RooMath::PositiveError,_nSigma);
  Double_t ym= sqrt(n), yp= ym;
  SetPoint(index,binCenter,n);
  SetPointError(index,0,0,-ym,+yp);
  if(n+yp > _ymax) _ymax= n+yp;
}

void RooHist::addAsymmetryBin(Float_t binCenter, Int_t n1, Int_t n2) {
}

RooHist::~RooHist() { }
