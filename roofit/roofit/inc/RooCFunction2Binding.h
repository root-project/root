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

#ifndef ROOCFUNCTION2BINDING
#define ROOCFUNCTION2BINDING

#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooFit/Detail/CFunctionBinding.h"

namespace RooFit {

typedef double (*CFUNCD2DD)(double,double) ;
typedef double (*CFUNCD2ID)(Int_t,double) ;
typedef double (*CFUNCD2UD)(UInt_t,double) ;
typedef double (*CFUNCD2DI)(double,Int_t) ;
typedef double (*CFUNCD2II)(Int_t,Int_t) ;


RooAbsReal* bindFunction(const char* name,CFUNCD2DD func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsReal* bindFunction(const char* name,CFUNCD2ID func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsReal* bindFunction(const char* name,CFUNCD2UD func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsReal* bindFunction(const char* name,CFUNCD2DI func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsReal* bindFunction(const char* name,CFUNCD2II func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD2DD func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD2ID func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD2UD func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD2DI func,RooAbsReal& x, RooAbsReal& y) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD2II func,RooAbsReal& x, RooAbsReal& y) ;

}

template <class VO, class VI1, class VI2>
using RooCFunction2Map = RooFit::Detail::CFunctionMap<VO (*)(VI1, VI2)>;



template<class VO, class VI1, class VI2>
class RooCFunction2Ref : public TObject {
 public:
  RooCFunction2Ref(VO (*ptr)(VI1,VI2)=nullptr) : _ptr(ptr) {
    // Constructor of persistable function reference
  } ;

  VO operator()(VI1 x,VI2 y) const {
    // Evaluate embedded function
    return (*_ptr)(x,y) ;
  }

  const char* name() const {
    // Return registered name of embedded function. If function
    // is not registered return string with hex presentation
    // of function pointer value
    return fmap().nameOrAddress(_ptr);
  }

  const char* argName(Int_t iarg) {
    // Return suggested name for i-th argument
    return fmap().lookupArgName(_ptr,iarg) ;
  }

  static RooCFunction2Map<VO,VI1,VI2>& fmap() {
    // Return reference to function pointer-to-name mapping service
    if (!_fmap) {
      _fmap = new RooCFunction2Map<VO,VI1,VI2> ;
    }
    return *_fmap ;
  }

 private:

  static VO dummyFunction(VI1,VI2) {
    // Dummy function used when registered function was not
    // found in un-persisting object
    return 0 ;
  }

  typedef VO (*func_t)(VI1,VI2);
  func_t _ptr; ///<! Pointer to embedded function

  static RooCFunction2Map<VO,VI1,VI2>* _fmap ; // Pointer to mapping service object

  ClassDefOverride(RooCFunction2Ref,1) // Persistable reference to C function pointer
} ;

// Define static member
template<class VO, class VI1, class VI2>
RooCFunction2Map<VO,VI1,VI2>* RooCFunction2Ref<VO,VI1,VI2>::_fmap = nullptr;


template<class VO, class VI1, class VI2>
void RooCFunction2Ref<VO,VI1,VI2>::Streamer(TBuffer &R__b)
{
   RooFit::Detail::streamCFunctionRef(R__b, *this, _ptr, dummyFunction, fmap());
}



template<class VO,class VI1, class VI2>
class RooCFunction2Binding : public RooAbsReal {
public:
  RooCFunction2Binding() {
    // Default constructor
  } ;
  RooCFunction2Binding(const char *name, const char *title, VO (*_func)(VI1,VI2), RooAbsReal& _x, RooAbsReal& _y);
  RooCFunction2Binding(const RooCFunction2Binding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction2Binding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction2Ref<VO,VI1,VI2> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x,y) ;
  }

private:

  ClassDefOverride(RooCFunction2Binding,1) // RooAbsReal binding to external C functions
};

template<class VO,class VI1, class VI2>
RooCFunction2Binding<VO,VI1,VI2>::RooCFunction2Binding(const char *name, const char *title, VO (*_func)(VI1,VI2),
                         RooAbsReal& _x, RooAbsReal& _y) :
  RooAbsReal(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x),
  y(func.argName(1),func.argName(1),this,_y)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI1, class VI2>
RooCFunction2Binding<VO,VI1,VI2>::RooCFunction2Binding(const RooCFunction2Binding& other, const char* name) :
  RooAbsReal(other,name),
  func(other.func),
  x("x",this,other.x),
  y("y",this,other.y)
{
  // Copy constructor
}




template<class VO,class VI1, class VI2>
class RooCFunction2PdfBinding : public RooAbsPdf {
public:
  RooCFunction2PdfBinding() {
    // Default constructor
  } ;
  RooCFunction2PdfBinding(const char *name, const char *title, VO (*_func)(VI1,VI2), RooAbsReal& _x, RooAbsReal& _y);
  RooCFunction2PdfBinding(const RooCFunction2PdfBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction2PdfBinding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction2Ref<VO,VI1,VI2> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x,y) ;
  }

private:

  ClassDefOverride(RooCFunction2PdfBinding,1) // RooAbsReal binding to external C functions
};

template<class VO,class VI1, class VI2>
RooCFunction2PdfBinding<VO,VI1,VI2>::RooCFunction2PdfBinding(const char *name, const char *title, VO (*_func)(VI1,VI2),
                         RooAbsReal& _x, RooAbsReal& _y) :
  RooAbsPdf(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x),
  y(func.argName(1),func.argName(1),this,_y)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI1, class VI2>
RooCFunction2PdfBinding<VO,VI1,VI2>::RooCFunction2PdfBinding(const RooCFunction2PdfBinding& other, const char* name) :
  RooAbsPdf(other,name),
  func(other.func),
  x("x",this,other.x),
  y("y",this,other.y)
{
  // Copy constructor
}

#endif
