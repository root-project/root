/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitBabar                                                      *
 *    File: $Id$
 * Authors:                                                                  *
 *    Aaron Roodman, Stanford Linear Accelerator Center, Stanford University *
 *                                                                           *
 * Copyright (c) 2004, Stanford University. All rights reserved.        *
 *           
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//
// The Parametric Step Function PDF is a binned distribution whose parameters 
// are the heights of each bin.  This PDF was first used in BaBar's B0->pi0pi0
// paper BABAR Collaboration (B. Aubert et al.) Phys.Rev.Lett.91:241801,2003.
// 
// This PDF may be used to describe oddly shaped distributions.  It differs
// from a RooKeysPdf or a RooHistPdf in that a RooParametricStepFunction
// has free parameters.  In particular, any statistical uncertainty in 
// sample used to model this PDF may be understood with these free parameters;
// this is not possible with non-parametric PDFs.
//
// The RooParametricStepFunction has Nbins-1 free parameters. Note that
// the limits of the dependent varaible must match the low and hi bin limits.
//
// An example of usage is:
//
// Int_t nbins(10);
// TArrayD limits(nbins+1);
// limits[0] = 0.0; //etc...
// RooArgList* list = new RooArgList("list");
// RooRealVar* binHeight0 = new RooRealVar("binHeight0","bin 0 Value",0.1,0.0,1.0);
// list->add(binHeight0); // up to binHeight8, ie. 9 parameters
//
// RooParametricStepFunction  aPdf = ("aPdf","PSF",*x,
//                                    *list,limits,nbins);
				

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooParametricStepFunction.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooParametricStepFunction)
;

RooParametricStepFunction::RooParametricStepFunction(const char* name, const char* title, 
			     RooAbsReal& x, const RooArgList& coefList, TArrayD& limits, Int_t nBins) :
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _nBins(nBins)
{
  // Constructor
  _coefIter = _coefList.createIterator() ;

  // Check lowest order
  if (_nBins<0) {
    cout << "RooParametricStepFunction::ctor(" << GetName() 
	 << ") WARNING: nBins must be >=0, setting value to 0" << endl ;
    _nBins=0 ;
  }

  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while(coef = (RooAbsArg*)coefIter->Next()) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooParametricStepFunction::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _coefList.add(*coef) ;
  }
  delete coefIter ;

  // Bin limits  
  limits.Copy(_limits);

}


RooParametricStepFunction::RooParametricStepFunction(const RooParametricStepFunction& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList),
  _nBins(other._nBins)
{
  // Copy constructor
  _coefIter = _coefList.createIterator();
  (other._limits).Copy(_limits);
}



RooParametricStepFunction::~RooParametricStepFunction()
{
  // Destructor
  delete _coefIter ;
}


Int_t RooParametricStepFunction::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}



Double_t RooParametricStepFunction::analyticalIntegral(Int_t code) const 
{
  assert(code==1) ;

  Double_t sum(1.0) ;
  return sum;  
  
}


Double_t RooParametricStepFunction::evaluate() const 
{
  Double_t xval(0.);
  xval = _x;
  Double_t value(0.);
  if (_x >= _limits[0] && _x < _limits[_nBins]){

    for (Int_t i=1;i<=_nBins;i++){
      if (_x < _limits[i]){
	// in Bin i-1 (starting with Bin 0)
	if (i<_nBins) {
	  // not in last Bin
	  RooRealVar* tmp = (RooRealVar*) _coefList.at(i-1);
	  value =  tmp->getVal();
	  break;
	} else {
	  // in last Bin
	  Double_t sum(0.);
	  Double_t binSize(0.);
	  for (Int_t j=1;j<_nBins;j++){
	    RooRealVar* tmp = (RooRealVar*) _coefList.at(j-1);
	    binSize = _limits[j] - _limits[j-1];
	    sum = sum + tmp->getVal()*binSize;
	  }
	  binSize = _limits[_nBins] - _limits[_nBins-1];
	  value = (1.0 - sum)/binSize;
	  if (value<=0.0){
	    value = 0.000001;
	    //	    cout << "RooParametricStepFunction: sum of values gt 1.0 -- beware!!" <<endl;
	  }
	  break;
	}
      } 
    }

  } 
  return value;

}

Int_t RooParametricStepFunction::getnBins(){
  return _nBins;
}

Double_t* RooParametricStepFunction::getLimits(){
  Double_t* limoutput = _limits.GetArray();
  return limoutput;
}
