/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDecay.cc,v 1.5 2001/11/05 18:53:48 verkerke Exp $
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
#include "RooFitCore/RooRandom.hh"

ClassImp(RooDecay) 
;


RooDecay::RooDecay(const char *name, const char *title, 
		   RooRealVar& t, RooAbsReal& tau, 
		   const RooResolutionModel& model, DecayType type) :
  RooConvolutedPdf(name,title,model,t), 
  _t("t","time",this,t),
  _tau("tau","decay time",this,tau),
  _type(type)
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
  _t("t",this,other._t),
  _tau("tau",this,other._tau),
  _basisExp(other._basisExp),
  _type(other._type)
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



Int_t RooDecay::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (matchArgs(directVars,generateVars,_t)) return 1 ;  
  return 0 ;
}



void RooDecay::generateEvent(Int_t code)
{
  assert(code==1) ;

  // Generate delta-t dependent
  while(1) {
    Double_t rand = RooRandom::uniform() ;
    Double_t tval(0) ;

    switch(_type) {
    case SingleSided:
      tval = -_tau*log(rand);
      break ;
    case Flipped:
      tval= +_tau*log(rand);
      break ;
    case DoubleSided:
      tval = (rand<=0.5) ? -_tau*log(2*rand) : +_tau*log(2*(rand-0.5)) ;
      break ;
    }
    
    if (tval<_t.max() && tval>_t.min()) {
      _t = tval ;
      break ;
    }
  }  
}
