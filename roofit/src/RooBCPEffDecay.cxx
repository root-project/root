/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBCPEffDecay.cc,v 1.1 2001/06/26 18:13:00 verkerke Exp $
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
  if (type==SingleSided || type==DoubleSided) 
    _basisExpPlus  = declareBasis("exp(-abs(@0)/@1)",RooArgSet(tau,dm)) ;

  if (type==Flipped || type==DoubleSided)
    _basisExpMinus = declareBasis("exp(-abs(-@0)/@1)",RooArgSet(tau,dm)) ;

  if (type==SingleSided || type==DoubleSided) 
    _basisSinPlus  = declareBasis("exp(-abs(@0)/@1)*sin(@0*@2)",RooArgSet(tau,dm)) ;

  if (type==Flipped || type==DoubleSided)
    _basisSinMinus = declareBasis("exp(-abs(-@0)/@1)*sin(@0*@2)",RooArgSet(tau,dm)) ;

  if (type==SingleSided || type==DoubleSided) 
    _basisCosPlus  = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgSet(tau,dm)) ;

  if (type==Flipped || type==DoubleSided)
    _basisCosMinus = declareBasis("exp(-abs(-@0)/@1)*cos(@0*@2)",RooArgSet(tau,dm)) ;
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
  _basisExpPlus(other._basisExpPlus),
  _basisExpMinus(other._basisExpMinus),
  _basisSinPlus(other._basisSinPlus),
  _basisSinMinus(other._basisSinMinus),
  _basisCosPlus(other._basisCosPlus),
  _basisCosMinus(other._basisCosMinus)
{
  // Copy constr4uctor
}



RooBCPEffDecay::~RooBCPEffDecay()
{
  // Destructor
}


Double_t RooBCPEffDecay::coefficient(Int_t basisIndex) const 
{
  if (basisIndex==_basisExpPlus || basisIndex==_basisExpMinus) {
    //exp term: (1+a^2)/2
    return (1+_absLambda*_absLambda)/2 ;
  }

  if (basisIndex==_basisSinPlus || basisIndex==_basisSinMinus) {
    //sin term: +/- (1-2w)*etaCP*a*b 
    return _tag*(1-2*_avgMistag)*_CPeigenval*_absLambda*_argLambda ;
  }
  
  if (basisIndex==_basisCosPlus || basisIndex==_basisCosMinus) {
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
    if (basisIndex==_basisExpPlus || basisIndex==_basisExpMinus) {
      return 2*coefficient(basisIndex) ;
    }
    
    if (basisIndex==_basisSinPlus || basisIndex==_basisSinMinus ||
        basisIndex==_basisCosPlus || basisIndex==_basisCosMinus)
      {
	return 0 ;
      }
  default:
    assert(0) ;
  }
    
  return 0 ;
}

