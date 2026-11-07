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

#ifndef ROOCFUNCTION1BINDING
#define ROOCFUNCTION1BINDING

#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooFit/Detail/CFunctionBinding.h"


namespace RooFit {

typedef double (*CFUNCD1D)(double) ;
typedef double (*CFUNCD1I)(Int_t) ;

RooAbsReal* bindFunction(const char* name,CFUNCD1D func,RooAbsReal& x) ;
RooAbsReal* bindFunction(const char* name,CFUNCD1I func,RooAbsReal& x) ;
RooAbsPdf*  bindPdf(const char* name,CFUNCD1D func,RooAbsReal& x) ;
RooAbsPdf*  bindPdf(const char* name,CFUNCD1I func,RooAbsReal& x) ;

}


template <class VO, class VI>
using RooCFunction1Map = RooFit::Detail::CFunctionMap<VO (*)(VI)>;



template<class VO, class VI>
class RooCFunction1Ref : public TObject {
 public:
  RooCFunction1Ref(VO (*ptr)(VI)=nullptr) : _ptr(ptr) {
    // Constructor of persistable function reference
  } ;

  VO operator()(VI x) const {
    // Evaluate embedded function
    return (*_ptr)(x) ;
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

 static RooCFunction1Map<VO,VI>& fmap();

 private:

  static VO dummyFunction(VI) {
    // Dummy function used when registered function was not
    // found in un-persisting object
    return 0 ;
  }

  typedef VO (*func_t)(VI);
  func_t _ptr; ///<! Pointer to embedded function

  static RooCFunction1Map<VO,VI>* _fmap ; // Pointer to mapping service object

  ClassDefOverride(RooCFunction1Ref,1) // Persistable reference to C function pointer
} ;

// Define static member
template<class VO, class VI>
RooCFunction1Map<VO,VI>* RooCFunction1Ref<VO,VI>::_fmap = nullptr;

template<class VO, class VI>
void RooCFunction1Ref<VO,VI>::Streamer(TBuffer &R__b)
{
   RooFit::Detail::streamCFunctionRef(R__b, *this, _ptr, dummyFunction, fmap());
}



template<class VO,class VI>
class RooCFunction1Binding : public RooAbsReal {
public:
  RooCFunction1Binding() {
    // Default constructor
  } ;
  RooCFunction1Binding(const char *name, const char *title, VO (*_func)(VI), RooAbsReal& _x);
  RooCFunction1Binding(const RooCFunction1Binding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction1Binding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction1Ref<VO,VI> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x) ;
  }

private:

  ClassDefOverride(RooCFunction1Binding,1) // RooAbsReal binding to external C functions
};


template<class VO,class VI>
RooCFunction1Binding<VO,VI>::RooCFunction1Binding(const char *name, const char *title, VO (*_func)(VI), RooAbsReal& _x) :
  RooAbsReal(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI>
RooCFunction1Binding<VO,VI>::RooCFunction1Binding(const RooCFunction1Binding& other, const char* name) :
  RooAbsReal(other,name),
  func(other.func),
  x("x",this,other.x)
{
  // Copy constructor
}



template<class VO,class VI>
class RooCFunction1PdfBinding : public RooAbsPdf {
public:
  RooCFunction1PdfBinding() {
    // Default constructor
  } ;
  RooCFunction1PdfBinding(const char *name, const char *title, VO (*_func)(VI), RooAbsReal& _x);
  RooCFunction1PdfBinding(const RooCFunction1PdfBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction1PdfBinding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction1Ref<VO,VI> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x) ;
  }

private:

  ClassDefOverride(RooCFunction1PdfBinding,1) // RooAbsReal binding to external C functions
};


template<class VO,class VI>
RooCFunction1PdfBinding<VO,VI>::RooCFunction1PdfBinding(const char *name, const char *title, VO (*_func)(VI), RooAbsReal& _x) :
  RooAbsPdf(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI>
RooCFunction1PdfBinding<VO,VI>::RooCFunction1PdfBinding(const RooCFunction1PdfBinding& other, const char* name) :
  RooAbsPdf(other,name),
  func(other.func),
  x("x",this,other.x)
{
  // Copy constructor
}

#endif
