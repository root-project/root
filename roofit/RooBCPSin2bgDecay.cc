/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooBCPSin2bgDecay.cc,v 1.1 2002/03/12 06:02:09 walkowia Exp $
 * Authors:
 *   WW, Wolfgang Walkowiak, UC Santa Cruz, walkowia@slac.stanford.edu
 * History:
 *   05-Mar-2002 WW Created initial version
 *   12-Mar-2002 WW Added alphaD0 and rhoD0
 *   13-Mar-2002 WW Added 2nd constructor for offset type scheme
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 
// This class provides two constructors for slightly different
// parametrizations of the same Pdf:
//
// 1)  f(dt,S_tag,S_mix) = 
//       gam*exp(-gam*|dt|)/(1+|Lambda|^2)/4 * [
//          ((1-S_tag*dw)*(1-alpha)+alpha*(1+S_mix*(1-2*rho)))*(1+|lambda|^2)
//         + S_mix*(1-2*w)*(1-alpha)*(1-|lambda|^2)*cos(dm*dt)
//         + S_tag*(1-2*w)*(1-alpha)*2*|lambda|*sin2bg*sin(dm*dt) ] 
//
// 2)  f(dt,S_tag,S_mix) = 
//       gam*exp(-gam*|dt|)/(1+|Lambda|^2)/4 * [
//          ( 1-S_tag*dw' + S_mix*offset )*(1+|lambda|^2)
//         + S_mix*(1-2*w')*(1-|lambda|^2)*cos(dm*dt)
//         + S_tag*(1-2*w')*2*|lambda|*sin2bg*sin(dm*dt) ] 
//
// with                                           / argLambdaPlus  for +
//     sin2bg = sin(2*beta+gamma+tag*mix*delta) = |
//                                                \\ argLambdaMinus for -
//
// Both parametrizations are equivalent, as can be easily seen using
//     w'     = w  * (1-alpha)+alpha/2
//     dw'    = dw * (1-alpha)
//     offset = alpha * (1-2*rho)
//

#include <iostream.h>
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealConstant.hh"
#include "RooFitCore/RooRandom.hh"
#include "RooFitModels/RooBCPSin2bgDecay.hh"

ClassImp(RooBCPSin2bgDecay) 
;

RooBCPSin2bgDecay::RooBCPSin2bgDecay(const char *name, const char *title, 
				     RooRealVar& t, RooAbsCategory& tag,
				     RooAbsCategory& mixState,
				     RooAbsReal& tau, RooAbsReal& dm,
				     RooAbsReal& avgMistag, 
				     RooAbsReal& a, RooAbsReal& bPlus,
				     RooAbsReal& bMinus, 
				     RooAbsReal& delMistag,
				     RooAbsReal& alphaD0, 
				     RooAbsReal& rhoD0,
				     const RooResolutionModel& model, 
				     DecayType type) :
    RooConvolutedPdf(name,title,model,t), 
    _absLambda("absLambda","Absolute value of lambda",this,a),
    _argLambdaPlus("argLambdaPlus","Arg(Lambda+)",this,bPlus),
    _argLambdaMinus("argLambdaPlus","Arg(Lambda-)",this,bMinus),
    _avgMistag("avgMistag","Average mistag rate",this,avgMistag),
    _delMistag("delMistag","Delta mistag rate",this,delMistag),  
    _alphaD0("alphaD0","D0 tagged fraction",this,alphaD0),
    _rhoOffD0("rhoOffD0","D0 mixed fraction",this,rhoD0),
    _tag("tag","Btag state",this,tag),
    _mixState("mixState","mix state",this,mixState),
    _tau("tau","decay time",this,tau),
    _dm("dm","mixing frequency",this,dm),
    _t("t","time",this,t),
    _type(type),
    _genFlavFrac(0),
    _genMixFrac(0),
    _genFlavFracMix(0),
    _genFlavFracUnmix(0),
    _offsetImp(kFALSE)
{
    // Constructor -- implementation 1 with alphaD0 and rhoD0
    chooseBase(type,tau,dm) ;
}

RooBCPSin2bgDecay::RooBCPSin2bgDecay(const char *name, const char *title, 
				     RooRealVar& t, RooAbsCategory& tag,
				     RooAbsCategory& mixState,
				     RooAbsReal& tau, RooAbsReal& dm,
				     RooAbsReal& avgMistag, 
				     RooAbsReal& a, RooAbsReal& bPlus,
				     RooAbsReal& bMinus, 
				     RooAbsReal& delMistag,
				     RooAbsReal& offsetD0, 
				     const RooResolutionModel& model, 
				     DecayType type) :
    RooConvolutedPdf(name,title,model,t), 
    _absLambda("absLambda","Absolute value of lambda",this,a),
    _argLambdaPlus("argLambdaPlus","Arg(Lambda+)",this,bPlus),
    _argLambdaMinus("argLambdaPlus","Arg(Lambda-)",this,bMinus),
    _avgMistag("avgMistag","Average mistag rate",this,avgMistag),
    _delMistag("delMistag","Delta mistag rate",this,delMistag),  
    _alphaD0("alphaD0","-- not used --",this,
	     (RooRealVar&)RooRealConstant::value(0.)),
    _rhoOffD0("rhoOffD0","D0 offset",this,offsetD0),
    _tag("tag","Btag state",this,tag),
    _mixState("mixState","mix state",this,mixState),
    _tau("tau","decay time",this,tau),
    _dm("dm","mixing frequency",this,dm),
    _t("t","time",this,t),
    _type(type),
    _genFlavFrac(0),
    _genMixFrac(0),
    _genFlavFracMix(0),
    _genFlavFracUnmix(0),
    _offsetImp(kTRUE)
{
    // Constructor -- implementation 2 with offset for decay-D0 tags
    chooseBase(type,tau,dm) ;
}

RooBCPSin2bgDecay::RooBCPSin2bgDecay(const RooBCPSin2bgDecay& other,
				     const char* name) : 
    RooConvolutedPdf(other,name), 
    _absLambda("absLambda",this,other._absLambda),
    _argLambdaPlus("argLambdaPlus",this,other._argLambdaPlus),
    _argLambdaMinus("argLambdaMinus",this,other._argLambdaMinus),
    _avgMistag("avgMistag",this,other._avgMistag),
    _delMistag("delMistag",this,other._delMistag),
    _alphaD0("alphaD0",this,other._alphaD0),
    _rhoOffD0("rhoOffD0",this,other._rhoOffD0),
    _tag("tag",this,other._tag),
    _mixState("mixState",this,other._mixState),
    _tau("tau",this,other._tau),
    _dm("dm",this,other._dm),
    _t("t",this,other._t),
    _type(other._type),
    _basisExp(other._basisExp),
    _basisSin(other._basisSin),
    _basisCos(other._basisCos),
    _genFlavFrac(other._genFlavFrac),
    _genMixFrac(other._genMixFrac),
    _genFlavFracMix(other._genFlavFracMix),
    _genFlavFracUnmix(other._genFlavFracUnmix),
    _offsetImp(other._offsetImp)
{
  // Copy constructor
}



RooBCPSin2bgDecay::~RooBCPSin2bgDecay()
{
  // Destructor
}


Double_t RooBCPSin2bgDecay::coefficient(Int_t basisIndex) const 
{
    // B0      : _tag      = +1 
    // B0bar   : _tag      = -1 
    // unmixed : _mixState = +1
    // mixed   : _mixState = -1

    Double_t dil, dw, off;
    calcInternals(dil,dw,off) ;

    if (basisIndex==_basisExp) {
	//exp term: (1 - tag*dw +mix*off)(1+a^2)/4 
	return (1 - _tag*dw + _mixState*off)*(1+_absLambda*_absLambda)/4 ;
    }
    
    if (basisIndex==_basisSin) {
	//sin term: tag*dil*absLambda*ImLambda/2
        // ( = tag * dil * |l| * sin(2b+g+/-d)/2 )
	if ( _tag*_mixState < 0 ) { 
	    return _tag*dil*_absLambda*_argLambdaMinus/2 ;
	} else {
	    return _tag*dil*_absLambda*_argLambdaPlus/2 ;
	}
    }
  
    if (basisIndex==_basisCos) {
	//cos term: mix * dil * (1-a^2)/4
	return _mixState*dil*(1-_absLambda*_absLambda)/4 ;
    } 
    
    return 0 ;
}



Int_t RooBCPSin2bgDecay::getCoefAnalyticalIntegral(RooArgSet& allVars, 
						    RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,_tag,_mixState)) return 3 ;
  if (matchArgs(allVars,analVars,_mixState)     ) return 2 ;
  if (matchArgs(allVars,analVars,_tag)          ) return 1 ;
  return 0 ;
}



Double_t RooBCPSin2bgDecay::coefAnalyticalIntegral(Int_t basisIndex, 
						    Int_t code) const 
{
    Double_t dil, dw, off;
    calcInternals(dil,dw,off) ;

    switch(code) {
	// No integration
	case 0: return coefficient(basisIndex) ;
	    
	    // integration over 'tag' and 'mixState'
	case 3: 
	    if (basisIndex==_basisExp) {
		return (1+_absLambda*_absLambda) ;
	    }
	    if (basisIndex==_basisSin || basisIndex==_basisCos) {
		return 0 ;
	    }
	    
	case 2:
	    // integration over 'mixState' 
	    if (basisIndex==_basisExp) {
		return (1+_absLambda*_absLambda)*(1-_tag*dw)/2 ;
	    }
	    if (basisIndex==_basisSin) {
		return _tag*dil*_absLambda
		    *(_argLambdaPlus+_argLambdaMinus)/2 ;
	    }
	    if (basisIndex==_basisCos) {
		return 0 ;
	    }

	    // Integration over 'tag'
	case 1:
	    if (basisIndex==_basisExp) {
		return (1+_absLambda*_absLambda)*(1+_mixState*off)/2 ;
	    }
	    if (basisIndex==_basisSin ) {
		return _mixState*dil*_absLambda
		*(_argLambdaPlus-_argLambdaMinus)/2 ;
	    }
	    if (basisIndex==_basisCos) {
		return _mixState*dil*(1-_absLambda*_absLambda)/2 ;
	    }
	    
	default:
	    assert(0) ;
    }
    
    return 0 ;
}



Int_t RooBCPSin2bgDecay::getGenerator(const RooArgSet& directVars, 
				       RooArgSet &generateVars) const
{
    if (matchArgs(directVars,generateVars,_t,_tag,_mixState)) return 4 ;  
    if (matchArgs(directVars,generateVars,_t,_mixState)     ) return 3 ;  
    if (matchArgs(directVars,generateVars,_t,_tag)          ) return 2 ;  
    if (matchArgs(directVars,generateVars,_t)               ) return 1 ;  
    return 0 ;
}



void RooBCPSin2bgDecay::initGenerator(Int_t code)
{
    switch (code) {
	case 2:
	{
	    // calculate the fraction of B0-tagged events to generate
	    Double_t sumInt = 
		RooRealIntegral("sumInt","sum integral",*this, 
				RooArgSet(_t.arg(),_tag.arg())).getVal();
	    _tag = 1 ; // B0
	    Double_t flavInt = 
		RooRealIntegral("flavInt","flavor integral",*this, 
				RooArgSet(_t.arg())).getVal();
	    _genFlavFrac = flavInt/sumInt;
	    break;
	}
	case 3:
	{
	    // calculate the fraction of mixed events to generate
	    Double_t sumInt = 
		RooRealIntegral("sumInt","sum integral",*this, 
				RooArgSet(_t.arg(),_mixState.arg())).getVal();
	    _mixState = -1 ; // mixed
	    Double_t mixInt = 
		RooRealIntegral("mixInt","mix integral",*this, 
				RooArgSet(_t.arg())).getVal();
	    _genMixFrac = mixInt/sumInt;
	    break;
	}
	case 4:
	{
	    // calculate the fraction of mixed events to generate
	    Double_t sumInt = 
		RooRealIntegral("sumInt","sum integral",*this, 
				RooArgSet(_t.arg(),_mixState.arg(),
					  _tag.arg())).getVal();
	    _mixState = -1 ; // mixed
	    Double_t mixInt = 
		RooRealIntegral("mixInt","mix integral",*this, 
				RooArgSet(_t.arg(),_tag.arg())).getVal();
	    _genMixFrac = mixInt/sumInt;

	    // calculate the fraction of B0 tagged for mixed and unmixed
	    // events to generate
	    RooRealIntegral dtInt("dtInt","dtintegral",*this, 
				  RooArgSet(_t.arg())) ;
	    _mixState = -1 ; // mixed
	    _tag      =  1 ; // B0
	    _genFlavFracMix   = dtInt.getVal() / mixInt ;
	    _mixState =  1 ; // unmixed
	    _tag      =  1 ; // B0
	    _genFlavFracUnmix = dtInt.getVal() / (sumInt - mixInt) ;
	    break;
	}
    }
}


void RooBCPSin2bgDecay::generateEvent(Int_t code)
{
    // Generate mix-state and/or tag-state dependent
    switch (code) { 
	case 2:
	{
	    Double_t rand = RooRandom::uniform() ;
	    _tag = (Int_t) ((rand<=_genFlavFrac) ? 1 : -1 );
	    break;
	}
	case 3:
	{
	    Double_t rand = RooRandom::uniform() ;
	    _mixState = (Int_t)( (rand<=_genMixFrac) ? -1 : 1) ;
	break;
	}
	case 4:
	{
	    Double_t rand = RooRandom::uniform() ;
	    _mixState =(Int_t) ( (rand<=_genMixFrac) ? -1 : 1) ;
		    
	    rand = RooRandom::uniform() ;
	    Double_t genFlavFrac = (_mixState==-1) ? _genFlavFracMix 
		: _genFlavFracUnmix ;
	    _tag = (Int_t) ((rand<=genFlavFrac) ? 1 : -1) ;
	    break;
	}
    }
    
    // Generate delta-t dependent
    Double_t dil, dw, off; 
    calcInternals(dil,dw,off) ;
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
		tval = 
		    (rand<=0.5) ? -_tau*log(2*rand) : +_tau*log(2*(rand-0.5)) ;
		break ;
	}

	// Accept event if T is in generated range
	Double_t al2 = _absLambda*_absLambda ;
	Double_t maxArgLambda = 
	    ( fabs(_argLambdaPlus)>fabs(_argLambdaMinus) ) ? 
	    fabs(_argLambdaPlus) : fabs(_argLambdaMinus) ;
	
	Double_t argLambda = 
	    (_mixState*_tag > 0) ? _argLambdaPlus : _argLambdaMinus ;
	
	Double_t maxAcceptProb = 
	    (1+fabs(dw)+fabs(off))*(1+al2) 
	    + fabs(dil*(1-al2)) 
	    + fabs(2*dil*_absLambda*maxArgLambda) ;

	Double_t acceptProb =   
	    (1-_tag*dw+_mixState*off)*(1+al2) 
	    + _mixState*dil*(1-al2)*cos(_dm*tval)
	    + _tag*2*dil*_absLambda*argLambda*sin(_dm*tval) ;

	// paranoid check
	if ( acceptProb > maxAcceptProb ) {
	    cout << "RooBCPSin2bgDecay::generateEvent: "
		 << "acceptProb > maxAcceptProb !!!" << endl;
	    assert(0);
	}
	
	Bool_t accept = 
	    maxAcceptProb*RooRandom::uniform() < acceptProb ? kTRUE : kFALSE ;
	
	if (tval<_t.max() && tval>_t.min() && accept) {
	    _t = tval ;
	    break ;
	}
    }
}

void RooBCPSin2bgDecay::calcInternals(Double_t& dil, Double_t& dw,
				      Double_t& off) const
{
    if ( _offsetImp ) {
	dil = 1-2*_avgMistag ;
	dw  = _delMistag ;
	off = _rhoOffD0 ;
    } else {
	dil = (1-2*_avgMistag)*(1-_alphaD0) ;
	dw  = _delMistag*(1-_alphaD0) ;
	off = _alphaD0*(1-2*_rhoOffD0) ;
    }
}

void RooBCPSin2bgDecay::chooseBase(DecayType type,
				   RooAbsReal& tau,
				   RooAbsReal& dm)
{
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


