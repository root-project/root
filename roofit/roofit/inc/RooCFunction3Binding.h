/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                   *   WV, Wouter Verkerke, NIKHEF, verkerke@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2008, NIKHEF, Regents of the University of California  *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 *****************************************************************************/

#ifndef ROOCFUNCTION3BINDING
#define ROOCFUNCTION3BINDING

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooAbsPdf.h"
#include "RooFit/Detail/CFunctionBinding.h"


namespace RooFit {

typedef double (*CFUNCD3DDD)(double,double,double) ;
typedef double (*CFUNCD3DDB)(double,double,bool) ;
typedef double (*CFUNCD3DII)(double,Int_t,Int_t) ;
typedef double (*CFUNCD3UDU)(UInt_t,double,UInt_t) ;
typedef double (*CFUNCD3UDD)(UInt_t,double,double) ;
typedef double (*CFUNCD3UUD)(UInt_t,UInt_t,double) ;

RooAbsReal* bindFunction(const char* name,CFUNCD3DDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsReal* bindFunction(const char* name,CFUNCD3DDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsReal* bindFunction(const char* name,CFUNCD3DII func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsReal* bindFunction(const char* name,CFUNCD3UDU func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsReal* bindFunction(const char* name,CFUNCD3UDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsReal* bindFunction(const char* name,CFUNCD3UUD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD3DDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD3DDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD3DII func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD3UDU func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD3UDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD3UUD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) ;

}

template <class VO, class VI1, class VI2, class VI3>
using RooCFunction3Map = RooFit::Detail::CFunctionMap<VO (*)(VI1, VI2, VI3)>;


template<class VO, class VI1, class VI2, class VI3>
class RooCFunction3Ref : public TObject {
 public:
  RooCFunction3Ref(VO (*ptr)(VI1,VI2,VI3)=nullptr) : _ptr(ptr) {
    // Constructor of persistable function reference
  } ;

  VO operator()(VI1 x,VI2 y, VI3 z) const {
    // Evaluate embedded function
    return (*_ptr)(x,y,z) ;
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

  static RooCFunction3Map<VO,VI1,VI2,VI3>& fmap() {
    // Return reference to function pointer-to-name mapping service
    if (!_fmap) {
      _fmap = new RooCFunction3Map<VO,VI1,VI2,VI3> ;
    }
    return *_fmap ;
  }

 private:

  static VO dummyFunction(VI1,VI2,VI3) {
    // Dummy function used when registered function was not
    // found in un-persisting object
    return 0 ;
  }


  typedef VO (*func_t)(VI1,VI2,VI3) ; ///<! Pointer to embedded function
  func_t _ptr; ///<! Pointer to embedded function

  static RooCFunction3Map<VO,VI1,VI2,VI3>* _fmap ; // Pointer to mapping service object

  ClassDefOverride(RooCFunction3Ref,1) // Persistable reference to C function pointer
} ;

// Define static member
template<class VO, class VI1, class VI2, class VI3>
RooCFunction3Map<VO,VI1,VI2,VI3>* RooCFunction3Ref<VO,VI1,VI2,VI3>::_fmap = nullptr;



template<class VO, class VI1, class VI2, class VI3>
void RooCFunction3Ref<VO,VI1,VI2,VI3>::Streamer(TBuffer &R__b)
{
   RooFit::Detail::streamCFunctionRef(R__b, *this, _ptr, dummyFunction, fmap());
}



template<class VO,class VI1, class VI2, class VI3>
class RooCFunction3Binding : public RooAbsReal {
public:
  RooCFunction3Binding() {
    // Default constructor
  } ;
  RooCFunction3Binding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3), RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z);
  RooCFunction3Binding(const RooCFunction3Binding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction3Binding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction3Ref<VO,VI1,VI2,VI3> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference
  RooRealProxy z ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x,y,z) ;
  }

private:

  ClassDefOverride(RooCFunction3Binding,1) // RooAbsReal binding to external C functions
};



template<class VO,class VI1, class VI2, class VI3>
RooCFunction3Binding<VO,VI1,VI2,VI3>::RooCFunction3Binding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3),
                         RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z) :
  RooAbsReal(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x),
  y(func.argName(1),func.argName(1),this,_y),
  z(func.argName(2),func.argName(2),this,_z)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI1, class VI2, class VI3>
RooCFunction3Binding<VO,VI1,VI2,VI3>::RooCFunction3Binding(const RooCFunction3Binding& other, const char* name) :
  RooAbsReal(other,name),
  func(other.func),
  x("x",this,other.x),
  y("y",this,other.y),
  z("z",this,other.z)
{
  // Copy constructor
}


template<class VO,class VI1, class VI2, class VI3>
class RooCFunction3PdfBinding : public RooAbsPdf {
public:
  RooCFunction3PdfBinding() {
    // Default constructor
  } ;
  RooCFunction3PdfBinding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3), RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z);
  RooCFunction3PdfBinding(const RooCFunction3PdfBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooCFunction3PdfBinding(*this,newname); }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    RooFit::Detail::printCFunctionArgs(os, *this, func.name());
  }

protected:

  RooCFunction3Ref<VO,VI1,VI2,VI3> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference
  RooRealProxy z ;              // Argument reference

  double evaluate() const override {
    // Return value of embedded function using value of referenced variable x
    return func(x,y,z) ;
  }

private:

  ClassDefOverride(RooCFunction3PdfBinding,1) // RooAbsReal binding to external C functions
};



template<class VO,class VI1, class VI2, class VI3>
RooCFunction3PdfBinding<VO,VI1,VI2,VI3>::RooCFunction3PdfBinding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3),
                         RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z) :
  RooAbsPdf(name,title),
  func(_func),
  x(func.argName(0),func.argName(0),this,_x),
  y(func.argName(1),func.argName(1),this,_y),
  z(func.argName(2),func.argName(2),this,_z)
{
  // Constructor of C function binding object given a pointer to a function and a RooRealVar to which the function
  // argument should be bound. This object is fully functional as a RooFit function object. The only restriction is
  // if the referenced function is _not_ a standard ROOT TMath or MathCore function it can not be persisted in a
  // a RooWorkspace
}


template<class VO,class VI1, class VI2, class VI3>
RooCFunction3PdfBinding<VO,VI1,VI2,VI3>::RooCFunction3PdfBinding(const RooCFunction3PdfBinding& other, const char* name) :
  RooAbsPdf(other,name),
  func(other.func),
  x("x",this,other.x),
  y("y",this,other.y),
  z("z",this,other.z)
{
  // Copy constructor
}

#endif
