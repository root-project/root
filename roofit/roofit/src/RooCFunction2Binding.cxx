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
// RooCFunction2Binding is a templated implementation of class RooAbsReal that binds 
// generic C(++) functions to a RooAbsReal argument thus allowing generic C++
// functions to be used as RooFit functions. Instances of function binding
// classes are fully functional RooFit function objects with one exception:
// if the bound function is _not_ a standard TMath or MathMore function the
// class cannot be persisted in a RooWorkspace without registering the function
// pointer first using RooCFunction2Binding<T1,T2,T3>::register().
// END_HTML
//

#include "Riostream.h" 
#include "RooCFunction2Binding.h" 
#include "RooCintUtils.h"

using namespace std ;

#ifndef ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
#define ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
templateClassImp(RooCFunction2Binding) 
templateClassImp(RooCFunction2Ref) 
#endif 


template<> RooCFunction2Map<double,double,double>* RooCFunction2Ref<double,double,double>::_fmap = 0 ;
template<> RooCFunction2Map<double,int,double> *RooCFunction2Ref<double,int,double>::_fmap = 0 ;
template<> RooCFunction2Map<double,unsigned int,double> *RooCFunction2Ref<double,unsigned int,double>::_fmap = 0 ;
template<> RooCFunction2Map<double,double,int> *RooCFunction2Ref<double,double,int>::_fmap = 0 ;
template<> RooCFunction2Map<double,int,int> *RooCFunction2Ref<double,int,int>::_fmap = 0 ;


namespace RooFit {

  RooAbsReal* bindFunction(const char* name,CFUNCD2DD func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2Binding<Double_t,Double_t,Double_t>(name,name,func,x,y) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD2ID func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2Binding<Double_t,Int_t,Double_t>(name,name,func,x,y) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD2UD func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2Binding<Double_t,UInt_t,Double_t>(name,name,func,x,y) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD2DI func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2Binding<Double_t,Double_t,Int_t>(name,name,func,x,y) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD2II func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2Binding<Double_t,Int_t,Int_t>(name,name,func,x,y) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD2DD func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2PdfBinding<Double_t,Double_t,Double_t>(name,name,func,x,y) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD2ID func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2PdfBinding<Double_t,Int_t,Double_t>(name,name,func,x,y) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD2UD func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2PdfBinding<Double_t,UInt_t,Double_t>(name,name,func,x,y) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD2DI func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2PdfBinding<Double_t,Double_t,Int_t>(name,name,func,x,y) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD2II func,RooAbsReal& x, RooAbsReal& y) {
    return new RooCFunction2PdfBinding<Double_t,Int_t,Int_t>(name,name,func,x,y) ;
  }

}
