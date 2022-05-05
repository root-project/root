/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooErrorVar.h,v 1.16 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ERROR_VAR
#define ROO_ERROR_VAR

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRealProxy.h"

#include <list>
#include <string>

class RooVectorDataStore;

class RooErrorVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  /// Default constructor
  inline RooErrorVar() {
  }
  RooErrorVar(const char *name, const char *title, const RooRealVar& input) ;
  RooErrorVar(const RooErrorVar& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooErrorVar(*this,newname); }
  ~RooErrorVar() override ;

  Double_t getValV(const RooArgSet* set=0) const override ;

  Double_t evaluate() const override {
    // return error of input RooRealVar
    return ((RooRealVar&)_realVar.arg()).getError() ;
  }

  void setVal(Double_t value) override {
    // Set error of input RooRealVar to value
    ((RooRealVar&)_realVar.arg()).setVal(value) ;
  }

  inline bool isFundamental() const override {
    // Return true as we implement a fundamental type of AbsArg that can be stored in a dataset
    return true ;
  }

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // Set/get finite fit range limits
  /// Set lower bound of default range to value
  inline void setMin(Double_t value) {
    setMin(0,value) ;
  }
  /// Set upper bound of default range to value
  inline void setMax(Double_t value) {
    setMax(0,value) ;
  }
  /// Set default ranges to [min,max]
  inline void setRange(Double_t min, Double_t max) {
    setRange(0,min,max) ;
  }
  void setMin(const char* name, Double_t value) ;
  void setMax(const char* name, Double_t value) ;
  void setRange(const char* name, Double_t min, Double_t max) ;

  void setBins(Int_t nBins);
  void setBinning(const RooAbsBinning& binning, const char* name=0) ;
  const RooAbsBinning& getBinning(const char* name=0, bool verbose=true, bool createOnTheFly=false) const override ;
  RooAbsBinning& getBinning(const char* name=0, bool verbose=true, bool createOnTheFly=false) override ;
  bool hasBinning(const char* name) const override ;
  std::list<std::string> getBinningNames() const override ;

  // Set infinite fit range limits
  void removeMin(const char* name=0);
  void removeMax(const char* name=0);
  void removeRange(const char* name=0);

  using RooAbsRealLValue::operator= ;
  using RooAbsRealLValue::setVal ;

protected:

  RooLinkedList _altBinning ;  ///<! Optional alternative ranges and binnings

  void syncCache(const RooArgSet* set=0) override ;

  RooRealProxy _realVar ;   ///< RealVar with the original error
  RooAbsBinning* _binning ; ///<! Pointer to default binning definition

  ClassDefOverride(RooErrorVar,1) // RooAbsRealLValue representation of an error of a RooRealVar
};

#endif
