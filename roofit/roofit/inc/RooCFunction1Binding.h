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
#include "RooMsgService.h"

#include "TBuffer.h"
#include "TString.h"

#include <string>
#include <map>
#include <vector>


namespace RooFit {

typedef double (*CFUNCD1D)(double) ;
typedef double (*CFUNCD1I)(Int_t) ;

RooAbsReal* bindFunction(const char* name,CFUNCD1D func,RooAbsReal& x) ;
RooAbsReal* bindFunction(const char* name,CFUNCD1I func,RooAbsReal& x) ;
RooAbsPdf*  bindPdf(const char* name,CFUNCD1D func,RooAbsReal& x) ;
RooAbsPdf*  bindPdf(const char* name,CFUNCD1I func,RooAbsReal& x) ;

}


template<class VO, class VI>
class RooCFunction1Map {
 public:
  RooCFunction1Map() {} ;

  void add(const char* name, VO (*ptr)(VI), const char* arg1name="x") {
    // Register function with given name and argument name
    _ptrmap[name] = ptr ;
    _namemap[ptr] = name ;
    _argnamemap[ptr].push_back(arg1name) ;
  }


  const char* lookupName(VO (*ptr)(VI)) {
    // Return name of function given by pointer
    return _namemap[ptr].c_str() ;
  }

  VO (*lookupPtr(const char* name))(VI) {
    // Return pointer of function given by name
    return _ptrmap[name] ;
  }

  const char* lookupArgName(VO (*ptr)(VI), UInt_t iarg) {
    // Return name of i-th argument of function. If function is
    // not registered, argument names 0,1,2 are x,y,z
    if (iarg<_argnamemap[ptr].size()) {
      return (_argnamemap[ptr])[iarg].c_str() ;
    }
    switch (iarg) {
    case 0: return "x" ;
    case 1: return "y" ;
    case 2: return "z" ;
    }
    return "w" ;
  }

 private:

#ifndef __CINT__
  std::map<std::string,VO (*)(VI)> _ptrmap ; // Pointer-to-name map
  std::map<VO (*)(VI),std::string> _namemap ; // Name-to-pointer map
  std::map<VO (*)(VI),std::vector<std::string> > _argnamemap ; // Pointer-to-argnamelist map
#endif
} ;



template<class VO, class VI>
class RooCFunction1Ref : public TObject {
 public:
  RooCFunction1Ref(VO (*ptr)(VI)=nullptr) : _ptr(ptr) {
    // Constructor of persistable function reference
  } ;
  ~RooCFunction1Ref() override {} ;

  VO operator()(VI x) const {
    // Evaluate embedded function
    return (*_ptr)(x) ;
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

 static RooCFunction1Map<VO,VI>& fmap();

 private:

  static VO dummyFunction(VI) {
    // Dummy function used when registered function was not
    // found in un-persisting object
    return 0 ;
  }

  typedef VO (*func_t)(VI);
  func_t _ptr; //! Pointer to embedded function

  static RooCFunction1Map<VO,VI>* _fmap ; // Pointer to mapping service object

  ClassDefOverride(RooCFunction1Ref,1) // Persistable reference to C function pointer
} ;

// Define static member
template<class VO, class VI>
RooCFunction1Map<VO,VI>* RooCFunction1Ref<VO,VI>::_fmap = 0;

template<class VO, class VI>
void RooCFunction1Ref<VO,VI>::Streamer(TBuffer &R__b)
{
  // Custom streamer for function pointer reference object. When writing,
  // the function pointer is substituted by its registered name. When function
  // is unregistered name 'UNKNOWN' is written and a warning is issues. When
  // reading back, the embedded name is converted back to a function pointer
  // using the mapping service. When name UNKNOWN is encountered a warning is
  // issues and a dummy null function is substituted. When the registered function
  // name can not be mapped to a function pointer an ERROR is issued and a pointer
  // to the dummy null function is substituted

  typedef ::RooCFunction1Ref<VO,VI> thisClass;

   // Stream an object of class RooCFunction1Ref
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

       // Lookup pointer to C function with given name
       _ptr = fmap().lookupPtr(tmpName.Data()) ;

       if (_ptr==nullptr) {
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
        // This union is to avoid a warning message:
        union {
           void *_ptr;
           func_t _funcptr;
        } temp;
        temp._funcptr = _ptr;
        coutW(ObjectHandling) << "WARNING: Cannot persist unknown function pointer " << Form("%p",temp._ptr)
                              << " written object will not be functional when read back" <<  std::endl ;
       tmpName="UNKNOWN" ;
     }

     // Persist the name
     tmpName.Streamer(R__b) ;

     R__b.SetByteCount(R__c, true);

   }
}



template<class VO,class VI>
class RooCFunction1Binding : public RooAbsReal {
public:
  RooCFunction1Binding() {
    // Default constructor
  } ;
  RooCFunction1Binding(const char *name, const char *title, VO (*_func)(VI), RooAbsReal& _x);
  RooCFunction1Binding(const RooCFunction1Binding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooCFunction1Binding(*this,newname); }
  inline ~RooCFunction1Binding() override { }

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
  TObject* clone(const char* newname) const override { return new RooCFunction1PdfBinding(*this,newname); }
  inline ~RooCFunction1PdfBinding() override { }

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
