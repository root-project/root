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
#include "RooMsgService.h"

#include "TBuffer.h"
#include "TString.h"

#include <string>
#include <map>
#include <vector>

namespace RooFit {

typedef Double_t (*CFUNCD4DDDD)(Double_t,Double_t,Double_t,Double_t) ;
typedef Double_t (*CFUNCD4DDDI)(Double_t,Double_t,Double_t,Int_t) ;
typedef Double_t (*CFUNCD4DDDB)(Double_t,Double_t,Double_t,bool) ;

RooAbsReal* bindFunction(const char* name,CFUNCD4DDDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsReal* bindFunction(const char* name,CFUNCD4DDDI func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsReal* bindFunction(const char* name,CFUNCD4DDDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDI func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;
RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) ;

}


template<class VO, class VI1, class VI2, class VI3, class VI4>
class RooCFunction4Map {
 public:
  RooCFunction4Map() {} ;

  void add(const char* name, VO (*ptr)(VI1,VI2,VI3,VI4), const char* arg1name="x", const char* arg2name="y",
                                                       const char* arg3name="z", const char* arg4name="w") {
    // Register function with given name and argument name
    _ptrmap[name] = ptr ;
    _namemap[ptr] = name ;
    _argnamemap[ptr].push_back(arg1name) ;
    _argnamemap[ptr].push_back(arg2name) ;
    _argnamemap[ptr].push_back(arg3name) ;
    _argnamemap[ptr].push_back(arg4name) ;
  }


  const char* lookupName(VO (*ptr)(VI1,VI2,VI3,VI4)) {
    // Return name of function given by pointer
    return _namemap[ptr].c_str() ;
  }

  VO (*lookupPtr(const char* name))(VI1,VI2,VI3,VI4) {
    // Return pointer of function given by name
    return _ptrmap[name] ;
  }

  const char* lookupArgName(VO (*ptr)(VI1,VI2,VI3,VI4), UInt_t iarg) {
    // Return name of i-th argument of function. If function is
    // not registered, argument names 0,1,2 are x,y,z
    if (iarg<_argnamemap[ptr].size()) {
      return (_argnamemap[ptr])[iarg].c_str() ;
    }
    switch (iarg) {
    case 0: return "x" ;
    case 1: return "y" ;
    case 2: return "z" ;
    case 3: return "w" ;
    }
    return "v" ;
  }

 private:

#ifndef __CINT__
  std::map<std::string,VO (*)(VI1,VI2,VI3,VI4)> _ptrmap ; // Pointer-to-name map
  std::map<VO (*)(VI1,VI2,VI3,VI4),std::string> _namemap ; // Name-to-pointer map
  std::map<VO (*)(VI1,VI2,VI3,VI4),std::vector<std::string> > _argnamemap ; // Pointer-to-argnamelist map
#endif
} ;



template<class VO, class VI1, class VI2, class VI3, class VI4>
class RooCFunction4Ref : public TObject {
 public:
  RooCFunction4Ref(VO (*ptr)(VI1,VI2,VI3,VI4)=0) : _ptr(ptr) {
    // Constructor of persistable function reference
  } ;
  ~RooCFunction4Ref() override {} ;

  VO operator()(VI1 x,VI2 y,VI3 z,VI4 w) const {
    // Evaluate embedded function
    return (*_ptr)(x,y,z,w) ;
  }

  const char* name() const {
    // Return registered name of embedded function. If function
    // is not registered return string with hex presentation
    // of function pointer value
    const char* result = fmap().lookupName(_ptr) ;
    if (result && strlen(result)) {
      return result ;
    }
    // This union is to avoid a warning message:
    union {
       void *_ptr;
       func_t _funcptr;
    } temp;
    temp._funcptr = _ptr;
    return Form("(%p)",temp._ptr) ;
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
  func_t _ptr; //! Pointer to embedded function

  static RooCFunction4Map<VO,VI1,VI2,VI3,VI4>* _fmap ; // Pointer to mapping service object

  ClassDefOverride(RooCFunction4Ref,1) // Persistable reference to C function pointer
} ;

// Define static member
template<class VO, class VI1, class VI2, class VI3, class VI4>
RooCFunction4Map<VO,VI1,VI2,VI3,VI4>* RooCFunction4Ref<VO,VI1,VI2,VI3,VI4>::_fmap = 0;

template<class VO, class VI1, class VI2, class VI3, class VI4>
void RooCFunction4Ref<VO,VI1,VI2,VI3,VI4>::Streamer(TBuffer &R__b)
{
  // Custom streamer for function pointer reference object. When writing,
  // the function pointer is substituted by its registerd name. When function
  // is unregistered name 'UNKNOWN' is written and a warning is issues. When
  // reading back, the embedded name is converted back to a function pointer
  // using the mapping service. When name UNKNOWN is encountered a warning is
  // issues and a dummy null function is substituted. When the registered function
  // name can not be mapped to a function pointer an ERROR is issued and a pointer
  // to the dummy null function is substituted

  typedef ::RooCFunction4Ref<VO,VI1,VI2,VI3,VI4> thisClass;

   // Stream an object of class RooCFunction4Ref
   if (R__b.IsReading()) {

     UInt_t R__s, R__c;
     Version_t R__v = R__b.ReadVersion(&R__s, &R__c);

     // Read name from file
     TString tmpName ;
     tmpName.Streamer(R__b) ;

     if (tmpName=="UNKNOWN" && R__v>0) {

       coutW(ObjectHandling) << "WARNING: Objected embeds function pointer to unknown function, object will not be functional" << std::endl ;
       _ptr = dummyFunction ;

     } else {

       // Lookup pointer to C function wih given name
       _ptr = fmap().lookupPtr(tmpName.Data()) ;

       if (_ptr==0) {
    coutW(ObjectHandling) << "ERROR: Objected embeds pointer to function named " << tmpName
                << " but no such function is registered, object will not be functional" << std::endl ;
       }
     }


     R__b.CheckByteCount(R__s, R__c, thisClass::IsA());

   } else {

     UInt_t R__c;
     R__c = R__b.WriteVersion(thisClass::IsA(), true);

     // Lookup name of reference C function
     TString tmpName = fmap().lookupName(_ptr) ;
     if (tmpName.Length()==0) {
       coutW(ObjectHandling) << "WARNING: Cannot persist unknown function pointer " << Form("0x%zx",(size_t)_ptr)
              << " written object will not be functional when read back" <<  std::endl ;
       tmpName="UNKNOWN" ;
     }

     // Persist the name
     tmpName.Streamer(R__b) ;

     R__b.SetByteCount(R__c, true);

   }
}



template<class VO,class VI1, class VI2, class VI3, class VI4>
class RooCFunction4Binding : public RooAbsReal {
public:
  RooCFunction4Binding() {
    // Default constructor
  } ;
  RooCFunction4Binding(const char *name, const char *title, VO (*_func)(VI1,VI2,VI3,VI4), RooAbsReal& _x, RooAbsReal& _y, RooAbsReal& _z, RooAbsReal& _w);
  RooCFunction4Binding(const RooCFunction4Binding& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooCFunction4Binding(*this,newname); }
  inline ~RooCFunction4Binding() override { }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    os << "[ function=" << func.name() << " " ;
    for (Int_t i=0 ; i<numProxies() ; i++) {
      RooAbsProxy* p = getProxy(i) ;
      if (!TString(p->name()).BeginsWith("!")) {
   p->print(os) ;
   os << " " ;
      }
    }
    os << "]" ;
  }

protected:

  RooCFunction4Ref<VO,VI1,VI2,VI3,VI4> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference
  RooRealProxy z ;              // Argument reference
  RooRealProxy w ;              // Argument reference

  Double_t evaluate() const override {
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
  RooCFunction4PdfBinding(const RooCFunction4PdfBinding& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooCFunction4PdfBinding(*this,newname); }
  inline ~RooCFunction4PdfBinding() override { }

  void printArgs(std::ostream& os) const override {
    // Print object arguments and name/address of function pointer
    os << "[ function=" << func.name() << " " ;
    for (Int_t i=0 ; i<numProxies() ; i++) {
      RooAbsProxy* p = getProxy(i) ;
      if (!TString(p->name()).BeginsWith("!")) {
   p->print(os) ;
   os << " " ;
      }
    }
    os << "]" ;
  }

protected:

  RooCFunction4Ref<VO,VI1,VI2,VI3,VI4> func ; // Function pointer reference
  RooRealProxy x ;              // Argument reference
  RooRealProxy y ;              // Argument reference
  RooRealProxy z ;              // Argument reference
  RooRealProxy w ;              // Argument reference

  Double_t evaluate() const override {
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
