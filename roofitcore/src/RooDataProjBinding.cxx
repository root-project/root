/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooDataProjBinding.cxx,v 1.18 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// Lightweight interface adaptor that projects a real function via summation
// of states provided in a dataset. The real function must be attached to
// the dataset before creating this binding object.
//
// If the dataset only contains category variables, the summation is optimized
// performing a weighted sum over the states of a RooSuperCategory that is
// constructed from all the categories in the dataset

#include "RooFit.h"

#include "RooDataProjBinding.h"
#include "RooDataProjBinding.h"
#include "RooAbsReal.h"
#include "RooAbsData.h"
#include "Roo1DTable.h"
#include "RooSuperCategory.h"
#include "RooCategory.h"
#include "RooAbsPdf.h"

#include <assert.h>

ClassImp(RooDataProjBinding)
;

RooDataProjBinding::RooDataProjBinding(const RooAbsReal &real, const RooAbsData& data, 
				       const RooArgSet &vars, const RooArgSet* nset) :
  RooRealBinding(real,vars,0), _first(kTRUE), _real(&real), _data(&data), _nset(nset), 
  _superCat(0), _catTable(0)
{  
  // Constructor

  // Determine if dataset contains only categories
  TIterator* iter = data.get()->createIterator() ;
  Bool_t allCat(kTRUE) ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooCategory*>(arg)) allCat = kFALSE ;
  }
  delete iter ;

  // Determine weights of various super categories fractions
  //allCat = kFALSE ;
  if (allCat) {
     _superCat = new RooSuperCategory("superCat","superCat",*data.get()) ;
     _catTable = data.table(*_superCat) ;
  }
}



RooDataProjBinding::~RooDataProjBinding() 
{
  // Destructor, delete owned objects
  if (_superCat) delete _superCat ;
  if (_catTable) delete _catTable ;
}



Double_t RooDataProjBinding::operator()(const Double_t xvector[]) const 
{
  // Evaluate data-projected values of boud real function.
  assert(isValid());
  loadValues(xvector);    

  //RooAbsArg::setDirtyInhibit(kTRUE) ;

  Double_t result(0) ;
  Double_t wgtSum(0) ;  

  if (_catTable) {

    // Data contains only categories, sum over weighted supercategory states
    TIterator* iter = _superCat->typeIterator() ;
    RooCatType* type ;
    while((type=(RooCatType*)iter->Next())) {
      // Backprop state to data set so that _real takes appropriate value
      _superCat->setIndex(type->getVal()) ;

      // Add weighted sum
      Double_t wgt = _catTable->get(type->GetName()) ;
      if (wgt) {
	result += wgt * _real->getVal(_nset) ;
	wgtSum += wgt ;
      }
    }
    delete iter ;
    
  } else {

    // Data contains reals, sum over all entries
    Int_t i ;
    Int_t nEvt = _data->numEntries() ;

    // Procedure might be lengthy, give some progress indication
    if (_first) {
      cout << "RooDataProjBinding::operator() projecting over " << nEvt << " events" << endl ;
      _first = kFALSE ;
    } else {
      cout << "." ; cout.flush() ;
    }

    for (i=0 ; i<nEvt ; i++) {
      _data->get(i) ;

      Double_t wgt = _data->weight() ;
      if (wgt) {
	result += wgt * _real->getVal(_nset) ;
	wgtSum += wgt ;
      }      
    }
  }

  //RooAbsArg::setDirtyInhibit(kFALSE) ;

  if (wgtSum==0) return 0 ;
  return result / wgtSum ;
}
