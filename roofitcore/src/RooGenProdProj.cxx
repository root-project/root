/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
//
// RooGenProdProj is an auxiliary class for RooProdPdf that calculates
// a general normalized projection of a product of non-factorizing PDFs, e.g.
//
//            Int ( P1 * P2 * ....) dx
//  P_x_xy = -------------------------------
//            Int (P1 * P2 * ... ) dx dy
//
// Partial integrals that factorize that can be calculated are calculated
// analytically. Remaining non-factorizing observables are integrated numerically.


#include <iostream.h>
#include <math.h>

#include "RooFitCore/RooGenProdProj.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooErrorHandler.hh"
#include "RooFitCore/RooProduct.hh"

ClassImp(RooGenProdProj)
;

RooGenProdProj::RooGenProdProj()
{
}


RooGenProdProj::RooGenProdProj(const char *name, const char *title, const RooArgSet& _prodSet, 
			       const RooArgSet& _intSet, const RooArgSet& _normSet) :
  RooAbsReal(name, title),
  _compSetN("compSetN","Set of integral components owned by numerator",this,kFALSE),
  _compSetD("compSetD","Set of integral components owned by denominator",this,kFALSE),
  _intList("intList","List of integrals",this,kTRUE),
  _compSetOwnedN(0), 
  _compSetOwnedD(0)
{
  // Constructor

  // Create owners of components created in ctor
  _compSetOwnedN = new RooArgSet ;
  _compSetOwnedD = new RooArgSet ;

  RooAbsReal* numerator = makeIntegral("numerator",_prodSet,_intSet,*_compSetOwnedN) ;
  RooAbsReal* denominator = makeIntegral("denominator",_prodSet,_normSet,*_compSetOwnedD) ;

  // Copy all components in (non-owning) set proxy
  _compSetN.add(*_compSetOwnedN) ;
  _compSetD.add(*_compSetOwnedD) ;
  
  _intList.add(*numerator) ;
  _intList.add(*denominator) ;
}



RooGenProdProj::RooGenProdProj(const RooGenProdProj& other, const char* name) :
  RooAbsReal(other, name), 
  _compSetN("compSetN","Set of integral components owned by numerator",this),
  _compSetD("compSetD","Set of integral components owned by denominator",this),
  _intList("intList","List of integrals",this),
  _compSetOwnedN(0), 
  _compSetOwnedD(0)
{
  // Copy constructor
  _compSetOwnedN = (RooArgSet*) other._compSetN.snapshot() ;
  _compSetN.add(*_compSetOwnedN) ;

  _compSetOwnedD = (RooArgSet*) other._compSetD.snapshot() ;
  _compSetD.add(*_compSetOwnedD) ;

  // Fill _intList
  _intList.add(*_compSetN.find(other._intList.at(0)->GetName())) ;
  _intList.add(*_compSetD.find(other._intList.at(1)->GetName())) ;
}


RooGenProdProj::~RooGenProdProj()
{
  if (_compSetOwnedN) delete _compSetOwnedN ;
  if (_compSetOwnedD) delete _compSetOwnedD ;
}



RooAbsReal* RooGenProdProj::makeIntegral(const char* name, const RooArgSet& compSet, const RooArgSet& intSet, RooArgSet& saveSet) 
{
  // Create integral of compSet over observables in intSet.
  RooArgSet anaIntSet, numIntSet ;

  // First determine subset of observables in intSet that are factorizable
  TIterator* compIter = compSet.createIterator() ;
  TIterator* intIter = intSet.createIterator() ;
  RooAbsPdf* pdf ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)intIter->Next()) {
    Int_t count(0) ;
    compIter->Reset() ;
    while(pdf=(RooAbsPdf*)compIter->Next()) {
      if (pdf->dependsOn(*arg)) count++ ;
    }

    if (count==0) {
    } else if (count==1) {
      anaIntSet.add(*arg) ;
    } else {
    }    
  }

  // Determine which of the factorizable integrals can be done analytically
  RooArgSet prodSet ;
  numIntSet.add(intSet) ;
  compIter->Reset() ;
  while(pdf=(RooAbsPdf*)compIter->Next()) {
    if (pdf->dependsOn(anaIntSet)) {
      RooArgSet anaSet ;
      Int_t code = pdf->getAnalyticalIntegralWN(anaIntSet,anaSet,0) ;
      if (code!=0) {
	// Analytical integral, create integral object
	RooAbsReal* pai = pdf->createIntegral(anaSet) ;

	// Add to integral to product
	prodSet.add(*pai) ;

	// Remove analytically integratable observables from numeric integration list
	numIntSet.remove(anaSet) ;

	// Declare ownership of integral
	saveSet.addOwned(*pai) ;
      } else {
	// Analytic integration of factorizable observable not possible, add straight pdf to product
	prodSet.add(*pdf) ;
      }      
    } else {
      // Non-factorizable observables, add straight pdf to product
      prodSet.add(*pdf) ;
    }
  }

  // Create product of (partial) analytical integrals
  RooProduct* prod = new RooProduct(Form("%s_%s",GetName(),name),"product",prodSet) ;

  // Declare owndership of product
  saveSet.addOwned(*prod) ;

  // Create integral performing remaining numeric integration over (partial) analytic product
  RooAbsReal* ret = prod->createIntegral(numIntSet) ;
  saveSet.addOwned(*ret) ;

  delete compIter ;
  delete intIter ;

  // Caller owners returned master integral object
  return ret ;
}


Double_t RooGenProdProj::evaluate() const 
{
  Double_t nom = ((RooAbsReal*)_intList.at(0))->getVal() ;
  Double_t den = ((RooAbsReal*)_intList.at(1))->getVal() ;
  return nom / den ;
}


