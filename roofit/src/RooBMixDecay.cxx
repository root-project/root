/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBMixDecay.cc,v 1.10 2001/11/21 07:01:28 verkerke Exp $
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
#include "RooFitCore/RooRealVar.hh"
#include "RooFitModels/RooBMixDecay.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooBMixDecay) 
;


RooBMixDecay::RooBMixDecay(const char *name, const char *title, 
			   RooRealVar& t, RooAbsCategory& mixState,
			   RooAbsCategory& tagFlav,
			   RooAbsReal& tau, RooAbsReal& dm,			   
			   RooAbsReal& mistag, RooAbsReal& delMistag,
			   const RooResolutionModel& model, 
			   DecayType type) :
  RooConvolutedPdf(name,title,model,t), 
  _mistag("mistag","Mistag rate",this,mistag),
  _mixState("mixState","Mixing state",this,mixState),
  _tagFlav("tagFlav","Flavour of tagged B0",this,tagFlav),
  _delMistag("delMistag","Delta mistag rate",this,delMistag),
  _type(type),
  _tau("tau","Mixing life time",this,tau),
  _dm("dm","Mixing frequency",this,dm),
  _t("_t","time",this,t), _genMixFrac(0)
{
  // Constructor
  switch(type) {
  case SingleSided:
    _basisExp = declareBasis("exp(-@0/@1)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case Flipped:
    _basisExp = declareBasis("exp(@0)/@1)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case DoubleSided:
    _basisExp = declareBasis("exp(-abs(@0)/@1)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  }
}


RooBMixDecay::RooBMixDecay(const RooBMixDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _mistag("mistag",this,other._mistag),
  _mixState("mixState",this,other._mixState),
  _tagFlav("tagFlav",this,other._tagFlav),
  _delMistag("delMistag",this,other._delMistag),
  _tau("tau",this,other._tau),
  _dm("dm",this,other._dm),
  _t("t",this,other._t),
  _basisExp(other._basisExp),
  _basisCos(other._basisCos),
  _type(other._type),
  _genMixFrac(other._genMixFrac),
  _genFlavFrac(other._genFlavFrac),
  _genFlavFracMix(other._genFlavFracMix),
  _genFlavFracUnmix(other._genFlavFracUnmix)
{
  // Copy constructor
}



RooBMixDecay::~RooBMixDecay()
{
  // Destructor
}


Double_t RooBMixDecay::coefficient(Int_t basisIndex) const 
{
  // Comp with tFit MC: must be (1 - tagFlav*...)
  if (basisIndex==_basisExp) {
    return (1 - _tagFlav*_delMistag) ; 
  }

  if (basisIndex==_basisCos) {
    return _mixState*(1-2*_mistag) ;   
  }
  
  return 0 ;
}



Int_t RooBMixDecay::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
//   cout << "RooBMixDecay::getCoefAI " ; allVars.Print("1") ;

  if (matchArgs(allVars,analVars,_mixState,_tagFlav)) return 3 ;
  if (matchArgs(allVars,analVars,_mixState)) return 2 ;
  if (matchArgs(allVars,analVars,_tagFlav)) return 1 ;
  return 0 ;
}



Double_t RooBMixDecay::coefAnalyticalIntegral(Int_t basisIndex, Int_t code) const 
{  
  switch(code) {
    // No integration
  case 0: return coefficient(basisIndex) ;

    // Integration over 'mixState' and 'tagFlav' 
  case 3:
    if (basisIndex==_basisExp) {
      return 4.0 ;
    }    
    if (basisIndex==_basisCos) {
      return 0.0 ;
    }

    // Integration over 'mixState'
  case 2:
    if (basisIndex==_basisExp) {
      return 2.0*coefficient(basisIndex) ;
    }    
    if (basisIndex==_basisCos) {
      return 0.0 ;
    }

    // Integration over 'tagFlav'
  case 1:
    if (basisIndex==_basisExp) {
      return 2.0 ;
    }    
    if (basisIndex==_basisCos) {
      return 2.0*coefficient(basisIndex) ;
    }
  default:
    assert(0) ;
  }
    
  return 0 ;
}


Int_t RooBMixDecay::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (staticInitOK) {
    if (matchArgs(directVars,generateVars,_t,_mixState,_tagFlav)) return 4 ;  
    if (matchArgs(directVars,generateVars,_t,_mixState)) return 3 ;  
    if (matchArgs(directVars,generateVars,_t,_tagFlav)) return 2 ;  
  }

  if (matchArgs(directVars,generateVars,_t)) return 1 ;  
  return 0 ;
}



void RooBMixDecay::initGenerator(Int_t code)
{
  switch (code) {
  case 2:
    {
      // Calculate the fraction of B0bar events to generate
      Double_t sumInt = RooRealIntegral("sumInt","sum integral",*this,RooArgSet(_t.arg(),_tagFlav.arg())).getVal() ;
      _tagFlav = 1 ; // B0 
      Double_t flavInt = RooRealIntegral("flavInt","flav integral",*this,RooArgSet(_t.arg())).getVal() ;
      _genFlavFrac = flavInt/sumInt ;
      break ;
    }  
  case 3:
    {
      // Calculate the fraction of mixed events to generate
      Double_t sumInt = RooRealIntegral("sumInt","sum integral",*this,RooArgSet(_t.arg(),_mixState.arg())).getVal() ;
      _mixState = -1 ; // mixed
      Double_t mixInt = RooRealIntegral("mixInt","mix integral",*this,RooArgSet(_t.arg())).getVal() ;
      _genMixFrac = mixInt/sumInt ;
      break ;
    }  
  case 4:
    {
      // Calculate the fraction of mixed events to generate
      Double_t sumInt = RooRealIntegral("sumInt","sum integral",*this,RooArgSet(_t.arg(),_mixState.arg(),_tagFlav.arg())).getVal() ;
      _mixState = -1 ; // mixed
      Double_t mixInt = RooRealIntegral("mixInt","mix integral",*this,RooArgSet(_t.arg(),_tagFlav.arg())).getVal() ;
      _genMixFrac = mixInt/sumInt ;
      
      // Calculate the fractio of B0bar tags for mixed and unmixed
      RooRealIntegral dtInt("mixInt","mix integral",*this,RooArgSet(_t.arg())) ;
      _mixState = -1 ; // Mixed
      _tagFlav  =  1 ; // B0
      _genFlavFracMix   = dtInt.getVal() / mixInt ;
      _mixState =  1 ; // Unmixed
      _tagFlav  =  1 ; // B0
      _genFlavFracUnmix = dtInt.getVal() / (sumInt - mixInt) ;
      break ;
    }
  }
}




void RooBMixDecay::generateEvent(Int_t code)
{
  // Generate mix-state dependent
  switch(code) {
  case 2:
    {
      Double_t rand = RooRandom::uniform() ;
      _tagFlav = (Int_t) ((rand<=_genFlavFrac) ?  1 : -1) ;
      break ;
    }
  case 3:
    {
      Double_t rand = RooRandom::uniform() ;
      _mixState = (Int_t) ((rand<=_genMixFrac) ? -1 : 1) ;
      break ;
    }
  case 4:
    {
      Double_t rand = RooRandom::uniform() ;
      _mixState = (Int_t) ((rand<=_genMixFrac) ? -1 : 1) ;

      rand = RooRandom::uniform() ;
      Double_t genFlavFrac = (_mixState==-1) ? _genFlavFracMix : _genFlavFracUnmix ;
      _tagFlav = (Int_t) ((rand<=genFlavFrac) ?  1 : -1) ;
      break ;
    }
  }

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

    // Accept event if T is in generated range
    Double_t dil = fabs(1-2*_mistag) ;
    Double_t maxAcceptProb = 1 + fabs(_delMistag) + dil ;
    Double_t acceptProb = (1-_tagFlav*_delMistag) + _mixState*dil*cos(_dm*tval);
    Bool_t mixAccept = maxAcceptProb*RooRandom::uniform() < acceptProb ? kTRUE : kFALSE ;
    
    if (tval<_t.max() && tval>_t.min() && mixAccept) {
      _t = tval ;
      break ;
    }
  }  
}
