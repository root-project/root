/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBCPEffDecay.cc,v 1.3 2001/10/27 22:32:28 verkerke Exp $
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
#include "RooFitModels/RooBCPEffDecay.hh"

ClassImp(RooBCPEffDecay) 
;

RooBCPEffDecay::RooBCPEffDecay(const char *name, const char *title, 
			       RooRealVar& t, RooAbsCategory& tag,
			       RooAbsReal& tau, RooAbsReal& dm,
			       RooAbsReal& avgMistag, RooAbsReal& CPeigenval,
			       RooAbsReal& a, RooRealVar& b,
			       RooAbsReal& effRatio, RooRealVar& delMistag,
			       const RooResolutionModel& model, DecayType type) :
  RooConvolutedPdf(name,title,model,t), 
  _absLambda("absLambda","Absolute value of lambda",this,a),
  _argLambda("argLambda","Arg(Lambda)",this,b),
  _CPeigenval("CPeigenval","CP eigen value",this,CPeigenval),
  _effRatio("effRatio","B0/B0bar efficiency ratio",this,effRatio),
  _avgMistag("avgMistag","Average mistag rate",this,avgMistag),
  _delMistag("delMistag","Delta mistag rate",this,delMistag),
  _tag("tag","CP state",this,tag)
{
  // Constructor
  switch(type) {
  case SingleSided:
    _basisExp = declareBasis("exp(-@0/@1)",RooArgList(tau,dm)) ;
    _basisSin = declareBasis("exp(-@0/@1)*sin(@0*@2)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case Flipped:
    _basisExp = declareBasis("exp(@0)/@1)",RooArgList(tau,dm)) ;
    _basisSin = declareBasis("exp(@0/@1)*sin(@0*@2)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case DoubleSided:
    _basisExp = declareBasis("exp(-abs(@0)/@1)",RooArgList(tau,dm)) ;
    _basisSin = declareBasis("exp(-abs(@0)/@1)*sin(@0*@2)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  }
}


RooBCPEffDecay::RooBCPEffDecay(const RooBCPEffDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _absLambda("absLambda",this,other._absLambda),
  _argLambda("argLambda",this,other._argLambda),
  _CPeigenval("CPeigenval",this,other._CPeigenval),
  _effRatio("effRatio",this,other._effRatio),
  _avgMistag("avgMistag",this,other._avgMistag),
  _delMistag("delMistag",this,other._delMistag),
  _tag("tag",this,other._tag),
  _basisExp(other._basisExp),
  _basisSin(other._basisSin),
  _basisCos(other._basisCos)
{
  // Copy constructor
}



RooBCPEffDecay::~RooBCPEffDecay()
{
  // Destructor
}


Double_t RooBCPEffDecay::coefficient(Int_t basisIndex) const 
{
  if (basisIndex==_basisExp) {
    //exp term: (1 -/+ dw)(1+a^2)/2
    return (1 - _tag*_delMistag)*(1+_absLambda*_absLambda)/2 ;
  }

  if (basisIndex==_basisSin) {
    //sin term: +/- (1-2w)*etaCP*a*b 
    return _tag*(1-2*_avgMistag)*_CPeigenval*_absLambda*_argLambda ;
  }
  
  if (basisIndex==_basisCos) {
    //cos term: +/- (1-2w)*(1-a^2)/2
    return _tag*(1-2*_avgMistag)*(1-_absLambda*_absLambda)/2 ;
  }
  
  return 0 ;
}



Int_t RooBCPEffDecay::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,_tag)) return 1 ;
  return 0 ;
}



Double_t RooBCPEffDecay::coefAnalyticalIntegral(Int_t basisIndex, Int_t code) const 
{
  switch(code) {
    // No integration
  case 0: return coefficient(basisIndex) ;

    // Integration over 'tag'
  case 1:
    if (basisIndex==_basisExp) {
      return (1+_absLambda*_absLambda) ;
    }
    
    if (basisIndex==_basisSin || _basisCos) {
      return 0 ;
    }
  default:
    assert(0) ;
  }
    
  return 0 ;
}

