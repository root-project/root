/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProduct.cc,v 1.9 2005/12/01 16:10:20 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
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


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooProduct.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooErrorHandler.h"

ClassImp(RooProduct)
;

RooProduct::RooProduct()
{
  _compRIter = _compRSet.createIterator() ;
  _compCIter = _compCSet.createIterator() ;
}


RooProduct::~RooProduct()
{
  delete _compRIter ;
  delete _compCIter ;
}


RooProduct::RooProduct(const char* name, const char* title, const RooArgSet& prodSet) :
  RooAbsReal(name, title),
  _compRSet("compRSet","Set of real product components",this),
  _compCSet("compCSet","Set of category product components",this)
{
  // Constructor
  _compRIter = _compRSet.createIterator() ;
  _compCIter = _compCSet.createIterator() ;


  TIterator* compIter = prodSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)compIter->Next())) {
    if (dynamic_cast<RooAbsReal*>(comp)) {
      _compRSet.add(*comp) ;
    } else if (dynamic_cast<RooAbsCategory*>(comp)) {
      _compCSet.add(*comp) ;
    } else {
      cout << "RooProduct::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
	   << " is not of type RooAbsReal or RooAbsCategory" << endl ;
      RooErrorHandler::softAbort() ;
    }
  }

  delete compIter ;
}



RooProduct::RooProduct(const RooProduct& other, const char* name) :
  RooAbsReal(other, name), 
  _compRSet("compRSet",this,other._compRSet),
  _compCSet("compCSet",this,other._compCSet)
{
  // Copy constructor
  _compRIter = _compRSet.createIterator() ;
  _compCIter = _compCSet.createIterator() ;
}




Double_t RooProduct::evaluate() const 
{
  Double_t prod(1) ;

  _compRIter->Reset() ;
  RooAbsReal* rcomp ;
  const RooArgSet* nset = _compRSet.nset() ;
  while((rcomp=(RooAbsReal*)_compRIter->Next())) {
    prod *= rcomp->getVal(nset) ;
  }
  
  _compCIter->Reset() ;
  RooAbsCategory* ccomp ;
  while((ccomp=(RooAbsCategory*)_compCIter->Next())) {
    prod *= ccomp->getIndex() ;
  }

  return prod ;
}

