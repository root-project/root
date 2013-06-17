/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, NIKHEF, verkerke@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2008, NIKHEF, Regents of the University of California  *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooCFunction1Binding is a templated implementation of class RooAbsReal that binds 
// generic C(++) functions to a RooAbsReal argument thus allowing generic C++
// functions to be used as RooFit functions. Instances of function binding
// classes are fully functional RooFit function objects with one exception:
// if the bound function is _not_ a standard TMath or MathMore function the
// class cannot be persisted in a RooWorkspace without registering the function
// pointer first using RooCFunction1Binding<T1,T2>::register().
// END_HTML
//

#include "Riostream.h" 
#include "RooCFunction1Binding.h" 
#include "RooCintUtils.h"

using namespace std ;

#ifndef ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
#define ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
templateClassImp(RooCFunction1Binding) 
templateClassImp(RooCFunction1Ref) 
#endif 


template<> RooCFunction1Map<double,double>* RooCFunction1Ref<double,double>::_fmap = 0 ;
template<> RooCFunction1Map<double,int>* RooCFunction1Ref<double,int>::_fmap = 0 ;

template<>
RooCFunction1Map<double,double>& RooCFunction1Ref<double,double>::fmap()
 {
    // Return reference to function pointer-to-name mapping service
    if (!_fmap) {
      _fmap = new RooCFunction1Map<double,double> ;
    }
    return *_fmap ;
  }

template<>
RooCFunction1Map<double,int>& RooCFunction1Ref<double,int>::fmap()
 {
    // Return reference to function pointer-to-name mapping service
    if (!_fmap) {
      _fmap = new RooCFunction1Map<double,int> ;
    }
    return *_fmap ;
 }

namespace RooFit {

  RooAbsReal* bindFunction(const char* name,CFUNCD1D func,RooAbsReal& x) {
    return new RooCFunction1Binding<Double_t,Double_t>(name,name,func,x) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD1I func,RooAbsReal& x) {
    return new RooCFunction1Binding<Double_t,Int_t>(name,name,func,x) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD1D func,RooAbsReal& x) {
    return new RooCFunction1PdfBinding<Double_t,Double_t>(name,name,func,x) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD1I func,RooAbsReal& x) {
    return new RooCFunction1PdfBinding<Double_t,Int_t>(name,name,func,x) ;
  }

}



