/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlotWithErrors.cc,v 1.2 2001/02/24 01:42:35 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

#include "RooFitCore/RooPlotWithErrors.hh"
//#include "RooFitTools/RooMath.hh"

ClassImp(RooPlotWithErrors)

static const char rcsid[] =
"$Id: RooPlotWithErrors.cc,v 1.2 2001/02/24 01:42:35 david Exp $";

RooPlotWithErrors::RooPlotWithErrors(Double_t nSigma) :
  TGraphAsymmErrors(), _nSigma(nSigma)
{
  initialize();
}

void RooPlotWithErrors::initialize() {
  _ymax= 0;
  SetMarkerStyle(8);
  SetDrawOption("P");
}

void RooPlotWithErrors::addBin(Float_t binCenter, Int_t n) {
  Int_t index= GetN();
//    Double_t ym= RooMath::PoissonError(n,RooMath::NegativeError,_nSigma);
//    Double_t yp= RooMath::PoissonError(n,RooMath::PositiveError,_nSigma);
  Double_t ym= sqrt(n), yp= ym;
  SetPoint(index,binCenter,n);
  SetPointError(index,0,0,-ym,+yp);
  if(n+yp > _ymax) _ymax= n+yp;
}

void RooPlotWithErrors::addAsymmetryBin(Float_t binCenter,
					Int_t n1, Int_t n2) {
}

RooPlotWithErrors::~RooPlotWithErrors() { }
