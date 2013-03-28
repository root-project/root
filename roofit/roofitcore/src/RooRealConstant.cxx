/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooRealConstant provides static functions to create and keep track
// of RooRealVar constants. Instead of creating such constants by
// hand (e.g. RooRealVar one("one","one",1)), simply use
// <pre>
//  RooRealConstant::value(1.0)
// </pre>
// whenever a reference to RooRealVar with constant value 1.0 is needed.
// RooRealConstant keeps an internal database of previously created
// RooRealVar objects and will recycle them as appropriate.
// END_HTML
//

#include "RooFit.h"

#include <math.h>
#include <sstream>
#include "RooRealConstant.h"
#include "RooRealConstant.h"
#include "RooConstVar.h"
#include "RooArgList.h"
#include "RooSentinel.h"

using namespace std;

ClassImp(RooRealConstant)
;


RooArgList* RooRealConstant::_constDB = 0;
TIterator* RooRealConstant::_constDBIter = 0;



//_____________________________________________________________________________
void RooRealConstant::cleanup() 
{
  // Cleanup function register with RooSentinel for cleanup in atexit()
  if (_constDB) {
    delete _constDB ;
    delete _constDBIter ;
    _constDB = 0 ;
  }
}



//_____________________________________________________________________________
RooConstVar& RooRealConstant::value(Double_t value) 
{
  // Return a constant value object with given value.
  // Return previously created object if available,
  // otherwise create a new one on the fly.
  
  // Lookup existing constant
  init() ;
  RooConstVar* var ;
  while((var=(RooConstVar*)_constDBIter->Next())) {
    if ((var->getVal()==value) && (!var->getAttribute("REMOVAL_DUMMY"))) return *var ;
  }

  // Create new constant
  std::ostringstream s ;
  s << value ;  

  var = new RooConstVar(s.str().c_str(),s.str().c_str(),value) ;
  var->setAttribute("RooRealConstant_Factory_Object",kTRUE) ;
  _constDB->addOwned(*var) ;

  return *var ;
}


//_____________________________________________________________________________
RooConstVar& RooRealConstant::removalDummy() 
{
  // Create a dummy node used in node-removal operations
  
  RooConstVar* var = new RooConstVar("REMOVAL_DUMMY","REMOVAL_DUMMY",1) ;
  var->setAttribute("RooRealConstant_Factory_Object",kTRUE) ;
  var->setAttribute("REMOVAL_DUMMY") ;
  _constDB->addOwned(*var) ;

  return *var ;
}



//_____________________________________________________________________________
void RooRealConstant::init() 
{
  // One-time initialization of constants database

  if (!_constDB) {
    _constDB = new RooArgList("RooRealVar Constants Database") ;
    _constDBIter = _constDB->createIterator() ;
    RooSentinel::activate() ;
  } else {
    _constDBIter->Reset() ;
  }
}
