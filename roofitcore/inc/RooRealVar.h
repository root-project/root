/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealVar.rdl,v 1.50 2005/12/01 16:10:20 wverkerke Exp $
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
#ifndef ROO_REAL_VAR
#define ROO_REAL_VAR

#include "Riostream.h"
#include <math.h>
#include <float.h>
#include "TString.h"

#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooUniformBinning.hh"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooSharedPropertiesList.hh"
#include "RooFitCore/RooRealVarSharedProperties.hh"

class RooArgSet ;
class RooErrorVar ;

class RooRealVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  RooRealVar() ;
  RooRealVar(const char *name, const char *title,
  	   Double_t value, const char *unit= "") ;
  RooRealVar(const char *name, const char *title, Double_t minValue, 
	   Double_t maxValue, const char *unit= "");
  RooRealVar(const char *name, const char *title, Double_t value, 
	   Double_t minValue, Double_t maxValue, const char *unit= "") ;
  RooRealVar(const RooRealVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooRealVar(*this,newname); }
  virtual ~RooRealVar();
  
  // Parameter value and error accessors
  virtual Double_t getVal(const RooArgSet* nset=0) const ;
  virtual void setVal(Double_t value);
  inline Double_t getError() const { return _error>=0?_error:0. ; }
  inline Bool_t hasError() const { return (_error>=0) ; }
  inline void setError(Double_t value) { _error= value ; }
  inline void removeError() { _error = -1 ; }
  inline Double_t getAsymErrorLo() const { return _asymErrLo<=0?_asymErrLo:0. ; }
  inline Double_t getAsymErrorHi() const { return _asymErrHi>=0?_asymErrHi:0. ; }
  inline Bool_t hasAsymError() const { return (_asymErrHi>=0 && _asymErrLo<=0) ; }
  inline void removeAsymError() { _asymErrLo = 1 ; _asymErrHi = -1 ; }
  inline void setAsymError(Double_t lo, Double_t hi) { _asymErrLo = lo ; _asymErrHi = hi ; }
  RooErrorVar* errorVar() const ;

  // Set/get finite fit range limits
  void setMin(const char* name, Double_t value) ;
  void setMax(const char* name, Double_t value) ;
  void setRange(const char* name, Double_t min, Double_t max) ;
  inline void setMin(Double_t value) { setMin(0,value) ; }
  inline void setMax(Double_t value) { setMax(0,value) ; }
  inline void setRange(Double_t min, Double_t max) { setRange(0,min,max) ; }

  void setBins(Int_t nBins) { setBinning(RooUniformBinning(getMin(),getMax(),nBins)) ; } 
  void setBinning(const RooAbsBinning& binning, const char* name=0) ;

  // RooAbsRealLValue implementation
  Bool_t hasBinning(const char* name) const ;
  const RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) const ;
  RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) ; 

  // Set infinite fit range limits
  inline void removeMin(const char* name=0) { getBinning(name).setMin(-RooNumber::infinity) ; }
  inline void removeMax(const char* name=0) { getBinning(name).setMax(RooNumber::infinity) ; }
  inline void removeRange(const char* name=0) { getBinning(name).setRange(-RooNumber::infinity,RooNumber::infinity) ; }
 
  // Backward compatibility functions
  void setFitBins(Int_t nBins) ;
  void setFitMin(Double_t value) ;
  void setFitMax(Double_t value) ;
  void setFitRange(Double_t min, Double_t max) ;
  void removeFitMin() ;
  void removeFitMax() ;
  void removeFitRange() ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline virtual Bool_t isFundamental() const { return kTRUE; }

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;


  TString* format(const RooCmdArg& formatArg) const ;
  TString* format(Int_t sigDigits, const char *options) const ;

  static void printScientific(Bool_t flag=kFALSE) { _printScientific = flag ; }
  static void printSigDigits(Int_t ndig=5) { _printSigDigits = ndig>1?ndig:1 ; }

  using RooAbsRealLValue::operator= ;

protected:

  static Bool_t _printScientific ;
  static Int_t  _printSigDigits ;

  virtual Double_t evaluate() const { return _value ; } // dummy because we overloaded getVal()
  virtual void copyCache(const RooAbsArg* source) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void fillTreeBranch(TTree& t) ;

  Double_t chopAt(Double_t what, Int_t where) const ;

  Double_t _error;      // Symmetric error associated with current value
  Double_t _asymErrLo ; // Low side of asymmetric error associated with current value
  Double_t _asymErrHi ; // High side of asymmetric error associated with current value
  RooAbsBinning* _binning ; 

  static RooSharedPropertiesList _sharedPropList; // List of properties shared among clone sets 
  RooRealVarSharedProperties* _sharedProp ; // Shared properties associated with this instance

  ClassDef(RooRealVar,3) // Real-valued variable 
};




#endif
