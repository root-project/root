/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDecay.cc,v 1.1 2001/06/08 05:52:38 verkerke Exp $
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
		   const RooResolutionModel& model, DecayType type) :
  RooConvolutedPdf(name,title,model,t)
{
  // Constructor
  if (type==SingleSided || type==DoubleSided) 
    _basisIdxPlus  = declareBasis("exp(-abs(@0)/@1)",tau) ;

  if (type==Flipped || type==DoubleSided)
    _basisIdxMinus = declareBasis("exp(-abs(-@0)/@1)",tau) ;
}


RooDecay::RooDecay(const RooDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _basisIdxPlus(other._basisIdxPlus),
  _basisIdxMinus(other._basisIdxMinus)
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


