/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDecay.cc,v 1.3 2001/10/08 05:21:16 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 

#include <iostream.h>
#include "RooFitModels/RooDecay.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooDecay) 
;


RooDecay::RooDecay(const char *name, const char *title, 
		   RooRealVar& t, RooAbsReal& tau, 
		   const RooResolutionModel& model, DecayType type) :
  RooConvolutedPdf(name,title,model,t)
{
  // Constructor
  switch(type) {
  case SingleSided:
    _basisExp = declareBasis("exp(-@0/@1)",tau) ;
    break ;
  case Flipped:
    _basisExp = declareBasis("exp(@0)/@1)",tau) ;
    break ;
  case DoubleSided:
    _basisExp = declareBasis("exp(-abs(@0)/@1)",tau) ;
    break ;
  }
}


RooDecay::RooDecay(const RooDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _basisExp(other._basisExp)
{
  // Copy constructor
}



RooDecay::~RooDecay()
{
  // Destructor
}


Double_t RooDecay::coefficient(Int_t basisIndex) const 
{
  return 1 ;
}


