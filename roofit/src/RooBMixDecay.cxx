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
#include "RooFitCore/RooRealVar.hh"
#include "RooFitModels/RooBMixDecay.hh"

ClassImp(RooBMixDecay) 
;


RooBMixDecay::RooBMixDecay(const char *name, const char *title, 
			   RooRealVar& t, RooAbsCategory& tag,
			   RooAbsReal& tau, RooAbsReal& dm,
			   RooAbsReal& mistag, const RooResolutionModel& model, 
			   DecayType type) :
  RooConvolutedPdf(name,title,model,t), 
  _mistag("mistag","Mistag rate",this,mistag),
  _tag("tag","Mixing state",this,tag)
{
  // Constructor
  if (type==SingleSided || type==DoubleSided) 
    _basisExpPlus  = declareBasis("exp(-abs(@0)/@1)",RooArgSet(tau,dm)) ;

  if (type==Flipped || type==DoubleSided)
    _basisExpMinus = declareBasis("exp(-abs(-@0)/@1)",RooArgSet(tau,dm)) ;

  if (type==SingleSided || type==DoubleSided) 
    _basisCosPlus  = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgSet(tau,dm)) ;

  if (type==Flipped || type==DoubleSided)
    _basisCosMinus = declareBasis("exp(-abs(-@0)/@1)*cos(@0*@2)",RooArgSet(tau,dm)) ;
}


RooBMixDecay::RooBMixDecay(const RooBMixDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _mistag("mistag",this,other._mistag),
  _tag("tag",this,other._tag),
  _basisExpPlus(other._basisExpPlus),
  _basisExpMinus(other._basisExpMinus),
  _basisCosPlus(other._basisCosPlus),
  _basisCosMinus(other._basisCosMinus)
{
  // Copy constructor
}



RooBMixDecay::~RooBMixDecay()
{
  // Destructor
}


Double_t RooBMixDecay::coefficient(Int_t basisIndex) const 
{
  if (basisIndex==_basisExpPlus || basisIndex==_basisExpMinus) {
    return 1 ;
  }

  if (basisIndex==_basisCosPlus || basisIndex==_basisCosMinus) {
    return _tag*(1-2*_mistag) ;
  }
  
  return 0 ;
}



Int_t RooBMixDecay::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,_tag)) return 1 ;
  return 0 ;
}



Double_t RooBMixDecay::coefAnalyticalIntegral(Int_t basisIndex, Int_t code) const 
{
  switch(code) {
    // No integration
  case 0: return coefficient(basisIndex) ;

    // Integration over 'tag'
  case 1:
    if (basisIndex==_basisExpPlus || basisIndex==_basisExpMinus) {
      return 2 ;
    }
    
    if (basisIndex==_basisCosPlus || basisIndex==_basisCosMinus) {
      return 0 ;
    }
  default:
    assert(0) ;
  }
    
  return 0 ;
}

