/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// 

#include <iostream.h>
#include "RooFitModels/RooDecay.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooDecay) 
;


RooDecay::RooDecay(const char *name, const char *title, 
		   RooRealVar& t, RooAbsReal& tau, 
		   const RooResolutionModel& model) :
  RooConvolutedPdf(name,title,model,t)
{
  // Constructor
  _basisIdx = declareBasis("exp(-abs(@0)/@1)",tau) ;
}


RooDecay::RooDecay(const RooDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _basisIdx(other._basisIdx)
{
  // Copy constructor
}



RooDecay::~RooDecay()
{
  // Destructor
}


Double_t RooDecay::coefficient(Int_t basisIndex) const 
{
  assert(basisIndex == _basisIdx) ;  
  return 1 ;
}


