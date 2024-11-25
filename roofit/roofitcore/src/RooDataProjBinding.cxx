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

/**
\file RooDataProjBinding.cxx
\class RooDataProjBinding
\ingroup Roofitcore

adaptor that projects a real function via summation of states
provided in a dataset. The real function must be attached to the
dataset before creating this binding object.

If the dataset only contains category variables, the summation is optimized
performing a weighted sum over the states of a RooSuperCategory that is
constructed from all the categories in the dataset

**/

#include "RooDataProjBinding.h"
#include "RooAbsReal.h"
#include "RooAbsData.h"
#include "Roo1DTable.h"
#include "RooSuperCategory.h"
#include "RooCategory.h"
#include "RooAbsPdf.h"
#include "RooMsgService.h"

#include <iostream>
#include <cassert>

using std::cout, std::endl;

ClassImp(RooDataProjBinding);

////////////////////////////////////////////////////////////////////////////////
/// Constructor of a data weighted average function binding with
/// variables 'vars' for function 'real' and dataset 'data' with
/// weights.

RooDataProjBinding::RooDataProjBinding(const RooAbsReal &real, const RooAbsData& data,
                   const RooArgSet &vars, const RooArgSet* nset) :
  RooRealBinding(real,vars,nullptr), _first(true), _real(&real), _data(&data), _nset(nset)
{
  // Determine if dataset contains only categories
  bool allCat(true) ;
  for(RooAbsArg * arg : *data.get()) {
    if (!dynamic_cast<RooCategory*>(arg)) allCat = false ;
  }

  // Determine weights of various super categories fractions
  if (allCat) {
     _superCat = std::make_unique<RooSuperCategory>("superCat","superCat",*data.get()) ;
     _catTable = std::unique_ptr<Roo1DTable>{data.table(*_superCat)};
  }
}

RooDataProjBinding::~RooDataProjBinding() = default;

////////////////////////////////////////////////////////////////////////////////
/// Evaluate data-projected values of the bound real function.

double RooDataProjBinding::operator()(const double xvector[]) const
{
  assert(isValid());
  loadValues(xvector);

  //RooAbsArg::setDirtyInhibit(true) ;

  double result(0) ;
  double wgtSum(0) ;

  if (_catTable) {

    // Data contains only categories, sum over weighted supercategory states
    for (const auto& nameIdx : *_superCat) {
      // Backprop state to data set so that _real takes appropriate value
      _superCat->setIndex(nameIdx) ;

      // Add weighted sum
      double wgt = _catTable->get(nameIdx.first.c_str());
      if (wgt) {
   result += wgt * _real->getVal(_nset) ;
   wgtSum += wgt ;
      }
    }

  } else {

    // Data contains reals, sum over all entries
    Int_t i ;
    Int_t nEvt = _data->numEntries() ;

    // Procedure might be lengthy, give some progress indication
    if (_first) {
      oocoutW(_real,Eval) << "RooDataProjBinding::operator() projecting over " << nEvt << " events" << endl ;
      _first = false ;
    } else {
      if (oodologW(_real,Eval)) {
   ooccoutW(_real,Eval) << "." ; cout.flush() ;
      }
    }

//     _real->Print("v") ;
//     ((RooAbsReal*)_real)->printCompactTree() ;

//     RooArgSet* params = _real->getObservables(_data->get()) ;

    for (i=0 ; i<nEvt ; i++) {
      _data->get(i) ;

      double wgt = _data->weight() ;
      double ret ;
      if (wgt) {
   ret = _real->getVal(_nset) ;
   result += wgt * ret ;
//    cout << "ret[" << i << "] = " ;
//    params->printStream(cout,RooPrintable::kName|RooPrintable::kValue,RooPrintable::kStandard) ;
//    cout << " = " << ret << endl ;
   wgtSum += wgt ;
      }
    }
  }

  //RooAbsArg::setDirtyInhibit(false) ;

  if (wgtSum==0) return 0 ;
  return result / wgtSum ;
}
