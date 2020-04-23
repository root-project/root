/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef __ROOFIT_NOROOMINIMIZER

//////////////////////////////////////////////////////////////////////////////
//
// RooMinimizerFcn is am interface class to the ROOT::Math function 
// for minization.
//                                                                                   

#include <iostream>

#include "RooFit.h"
#include "RooMinimizerFcn.h"

#include "Riostream.h"

#include "TIterator.h"
#include "TClass.h"

#include "RooAbsArg.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"

#include "RooMinimizer.h"
#include "RooGaussMinimizer.h"

using namespace std;

RooMinimizerFcn::RooMinimizerFcn(RooAbsReal *funct, RooMinimizer* context,
			   bool verbose) :
  RooAbsMinimizerFcn(*funct->getParameters(RooArgSet()), context, verbose), _funct(funct)
{}



RooMinimizerFcn::RooMinimizerFcn(const RooMinimizerFcn& other) : RooAbsMinimizerFcn(other), ROOT::Math::IBaseFunctionMultiDim(other),
  _funct(other._funct)
{}


RooMinimizerFcn::~RooMinimizerFcn()
{}


ROOT::Math::IBaseFunctionMultiDim* RooMinimizerFcn::Clone() const 
{  
  return new RooMinimizerFcn(*this) ;
}


void RooMinimizerFcn::optimizeConstantTerms(bool constStatChange, bool constValChange) {
   if (constStatChange) {

      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;

      oocoutI(_context,Minimization) << "RooMinimizerFcn::synchronize: set of constant parameters changed, rerunning const optimizer" << endl ;
      _funct->constOptimizeTestStatistic(RooAbsArg::ConfigChange) ;
   } else if (constValChange) {
      oocoutI(_context,Minimization) << "RooMinimizerFcn::synchronize: constant parameter values changed, rerunning const optimizer" << endl ;
      _funct->constOptimizeTestStatistic(RooAbsArg::ValueChange) ;
   }

   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
}


double RooMinimizerFcn::DoEval(const double *x) const 
{

  // Set the parameter values for this iteration
  for (int index = 0; index < _nDim; index++) {
    if (_logfile) (*_logfile) << x[index] << " " ;
    SetPdfParamVal(index,x[index]);
  }

  // Calculate the function for these parameters  
  RooAbsReal::setHideOffset(kFALSE) ;
  double fvalue = _funct->getVal();
  RooAbsReal::setHideOffset(kTRUE) ;

  if (RooAbsPdf::evalError() || RooAbsReal::numEvalErrors()>0 || fvalue>1e30) {

    if (_printEvalErrors>=0) {

      if (_doEvalErrorWall) {
        oocoutW(_context,Minimization) << "RooMinimizerFcn: Minimized function has error status." << endl 
				       << "Returning maximum FCN so far (" << _maxFCN 
				       << ") to force MIGRAD to back out of this region. Error log follows" << endl ;
      } else {
        oocoutW(_context,Minimization) << "RooMinimizerFcn: Minimized function has error status but is ignored" << endl ;
      } 

      TIterator* iter = _floatParamList->createIterator() ;
      RooRealVar* var ;
      Bool_t first(kTRUE) ;
      ooccoutW(_context,Minimization) << "Parameter values: " ;
      while((var=(RooRealVar*)iter->Next())) {
        if (first) { first = kFALSE ; } else ooccoutW(_context,Minimization) << ", " ;
        ooccoutW(_context,Minimization) << var->GetName() << "=" << var->getVal() ;
      }
      delete iter ;
      ooccoutW(_context,Minimization) << endl ;
      
      RooAbsReal::printEvalErrors(ooccoutW(_context,Minimization),_printEvalErrors) ;
      ooccoutW(_context,Minimization) << endl ;
    } 

    if (_doEvalErrorWall) {
      fvalue = _maxFCN+1 ;
    }

    RooAbsPdf::clearEvalError() ;
    RooAbsReal::clearEvalErrorLog() ;
    _numBadNLL++ ;
  } else if (fvalue>_maxFCN) {
    _maxFCN = fvalue ;
  }
      
  // Optional logging
  if (_logfile) 
    (*_logfile) << setprecision(15) << fvalue << setprecision(4) << endl;
  if (_verbose) {
    cout << "\nprevFCN" << (_funct->isOffsetting()?"-offset":"") << " = " << setprecision(10) 
         << fvalue << setprecision(4) << "  " ;
    cout.flush() ;
  }

  _evalCounter++ ;

  return fvalue;
}

#endif

