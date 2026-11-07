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

#ifndef ROOCFUNCTION4BINDING
#define ROOCFUNCTION4BINDING

#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooFit/Detail/CFunctionBinding.h"

namespace RooFit {

typedef double (*CFUNCD4DDDD)(double,double,double,double) ;
typedef double (*CFUNCD4DDDI)(double,double,double,Int_t) ;
typedef double (*CFUNCD4DDDB)(double,double,double,bool) ;

RooAbsReal* bindFunction(const char* name,CFUNCD4DDDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsReal* bindFunction(const char* name,CFUNCD4DDDI func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsReal* bindFunction(const char* name,CFUNCD4DDDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDI func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;

}


template <class VO, class VI1, class VI2, class VI3, class VI4>
using RooCFunction4Map = RooFit::Detail::CFunctionMap<VO (*)(VI1, VI2, VI3, VI4)>;



template<class VO, class VI1, class VI2, class VI3, class VI4>
class RooCFunction4Ref : public TObject {
 public:
  RooCFunction4Ref(VO (*ptr)(VI1,VI2,VI3,VI4)=nullptr) : _ptr(ptr) {
    // Constructor of persistable function reference
  } ;

  VO operator()(VI1 x,VI2 y,VI3 z,VI4 w) const {
    // Evaluate embedded function
    return (*_ptr)(x,y,z,w) ;
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

  static RooCFunction4Map<VO,VI1,VI2,VI3,VI4>& fmap() {
    // Return reference to function pointer-to-name mapping service
    if (!_fmap) {
      _fmap = new RooCFunction4Map<VO,VI1,VI2,VI3,VI4> ;
    }
    return *_fmap ;
  }
 private:

  static VO dummyFunction(VI1,VI2,VI3,VI4) {
    // Dummy function used when registered function was not
    // found in un-persisting object
    return 0 ;
  }


  typedef VO (*func_t)(VI1,VI2,VI3,VI4);
  func_t _ptr; ///<! Pointer to embedded function

  static RooCFunction4Map<VO,VI1,VI2,VI3,VI4>* _fmap ; // Pointer to mapping service object

  ClassDefOverride(RooCFunction4Ref,1) // Persistable reference to C function pointer
} ;

// Define static member
template<class VO, class VI1, class VI2, class VI3, class VI4>
RooCFunction4Map<VO,VI1,VI2,VI3,VI4>* RooCFunction4Ref<VO,VI1,VI2,VI3,VI4>::_fmap = nullptr;

template<class VO, class VI1, class VI2, class VI3, class VI4>
void RooCFunction4Ref<VO,VI1,VI2,VI3,VI4>::Streamer(TBuffer &R__b)
{
   RooFit::Detail::streamCFunctionRef(R__b, *this, _ptr, dummyFunction, fmap());
}



template<class VO,class VI1, class VI2, class VI3, class VI4>
class RooCFunction4Binding : public RooAbsReal {
public:
  RooCFunction4Binding() {
    // Default constructor
  } ;
  RooCFunction4Binding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3,VI4), RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z, RooAbsReal& _w);
  RooCFunction4Binding(const RooCFunction4Binding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction4Binding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction4Ref<VO,VI1,VI2,VI3,VI4> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference
  RooRealProxy z ;              // Argument reference
  RooRealProxy w ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x,y,z,w) ;
  }

private:

  ClassDefOverride(RooCFunction4Binding,1) // RooAbsReal binding to external C functions
};


template<class VO,class VI1, class VI2, class VI3, class VI4>
RooCFunction4Binding<VO,VI1,VI2,VI3,VI4>::RooCFunction4Binding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3,VI4),
                         RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z, RooAbsReal& _w) :
  RooAbsReal(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x),
  y(func.argName(1),func.argName(1),this,_y),
  z(func.argName(2),func.argName(2),this,_z),
  w(func.argName(3),func.argName(3),this,_w)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI1, class VI2, class VI3, class VI4>
RooCFunction4Binding<VO,VI1,VI2,VI3,VI4>::RooCFunction4Binding(const RooCFunction4Binding& other, const char* name) :
  RooAbsReal(other,name),
  func(other.func),
  x("x",this,other.x),
  y("y",this,other.y),
  z("z",this,other.z),
  w("w",this,other.w)
{
  // Copy constructor
}


template<class VO,class VI1, class VI2, class VI3, class VI4>
class RooCFunction4PdfBinding : public RooAbsPdf {
public:
  RooCFunction4PdfBinding() {
    // Default constructor
  } ;
  RooCFunction4PdfBinding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3,VI4), RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z, RooAbsReal& _w);
  RooCFunction4PdfBinding(const RooCFunction4PdfBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction4PdfBinding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction4Ref<VO,VI1,VI2,VI3,VI4> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference
  RooRealProxy z ;              // Argument reference
  RooRealProxy w ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x,y,z,w) ;
  }

private:

  ClassDefOverride(RooCFunction4PdfBinding,1) // RooAbsReal binding to external C functions
};


template<class VO,class VI1, class VI2, class VI3, class VI4>
RooCFunction4PdfBinding<VO,VI1,VI2,VI3,VI4>::RooCFunction4PdfBinding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3,VI4),
                         RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z, RooAbsReal& _w) :
  RooAbsPdf(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x),
  y(func.argName(1),func.argName(1),this,_y),
  z(func.argName(2),func.argName(2),this,_z),
  w(func.argName(3),func.argName(3),this,_w)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI1, class VI2, class VI3, class VI4>
RooCFunction4PdfBinding<VO,VI1,VI2,VI3,VI4>::RooCFunction4PdfBinding(const RooCFunction4PdfBinding& other, const char* name) :
  RooAbsPdf(other,name),
  func(other.func),
  x("x",this,other.x),
  y("y",this,other.y),
  z("z",this,other.z),
  w("w",this,other.w)
{
  // Copy constructor
}

#endif
