/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProduct.cc,v 1.1 2003/04/28 20:42:41 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
//
// RooProduct calculates the product of a set of RooAbsReal terms.
// This class does not (yet) do any smart handling of integrals, i.e
// all integrals of the product are handled numerically


#include <iostream.h>
#include <math.h>

#include "RooFitCore/RooProduct.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooErrorHandler.hh"

ClassImp(RooProduct)
;

RooProduct::RooProduct()
{
  _compIter = _compSet.createIterator() ;
}


RooProduct::RooProduct(const char* name, const char* title, const RooArgSet& prodSet) :
  RooAbsReal(name, title),
  _compSet("compSet","Set of product components",this)
{
  // Constructor
  _compIter = _compSet.createIterator() ;


  TIterator* compIter = prodSet.createIterator() ;
  RooAbsArg* comp ;
  while(comp = (RooAbsArg*)compIter->Next()) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      cout << "RooProduct::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _compSet.add(*comp) ;
  }

  delete compIter ;
}



RooProduct::RooProduct(const RooProduct& other, const char* name) :
  RooAbsReal(other, name), 
  _compSet("compSet",this,other._compSet)
{
  // Copy constructor
  _compIter = _compSet.createIterator() ;
}




Double_t RooProduct::evaluate() const 
{
  Double_t prod(1) ;
  _compIter->Reset() ;

  RooAbsReal* comp ;
  const RooArgSet* nset = _compSet.nset() ;
  while(comp=(RooAbsReal*)_compIter->Next()) {
    prod *= comp->getVal(nset) ;
  }
  
  return prod ;
}


