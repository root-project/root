/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCategory.rdl,v 1.37 2005/12/08 13:19:54 wverkerke Exp $
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
#ifndef ROO_ABS_CATEGORY
#define ROO_ABS_CATEGORY

#include "Riostream.h"
#include "TObjArray.h"
#include "RooAbsArg.h"
#include "RooCatType.h"

class TTree ;
class RooArgSet ;
class RooDataSet ;
class Roo1DTable ;

class RooAbsCategory : public RooAbsArg {
public:
  // Constructors, assignment etc.
  RooAbsCategory() { _typeIter = _types.MakeIterator() ; } ;
  RooAbsCategory(const char *name, const char *title);
  RooAbsCategory(const RooAbsCategory& other, const char* name=0) ;
  virtual ~RooAbsCategory();
  
  // Value accessors
  virtual Int_t getIndex() const ;
  virtual const char* getLabel() const ;
  Bool_t operator==(Int_t index) const ;
  Bool_t operator!=(Int_t index) { return !operator==(index);}
  Bool_t operator==(const char* label) const ;
  Bool_t operator!=(const char* label) { return !operator==(label);}
  virtual Bool_t operator==(const RooAbsArg& other) ;
  Bool_t         operator!=(const RooAbsArg& other) { return !operator==(other);}
  
  Bool_t isValidIndex(Int_t index) const ;
  Bool_t isValidLabel(const char* label) const ;  
  const RooCatType* lookupType(Int_t index, Bool_t printError=kFALSE) const ;
  const RooCatType* lookupType(const char* label, Bool_t printError=kFALSE) const ;
  const RooCatType* lookupType(const RooCatType& type, Bool_t printError=kFALSE) const ;
  TIterator* typeIterator() const ;
  Int_t numTypes() const { return _types.GetEntries() ; }
  Bool_t isSignType(Bool_t mustHaveZero=kFALSE) const ;

  Roo1DTable *createTable(const char *label) const ;

  // I/O streaming interface
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  virtual Bool_t isIntegrationSafeLValue(const RooArgSet* /*set*/) const { return kTRUE ; }

  RooAbsArg *createFundamental(const char* newname=0) const;

protected:

  // Function evaluation and error tracing
  RooCatType traceEval() const ;
  virtual Bool_t traceEvalHook(RooCatType /*value*/) const { return kFALSE ;}
  virtual RooCatType evaluate() const = 0 ;

  // Type definition management
  const RooCatType* defineType(const char* label) ;
  const RooCatType* defineType(const char* label, Int_t index) ;
  const RooCatType* defineTypeUnchecked(const char* label, Int_t index) ;
  const RooCatType* getOrdinal(UInt_t n) const;
  void clearTypes() ;

  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(RooCatType value) const ;

  virtual void syncCache(const RooArgSet* set=0) ;
  virtual void copyCache(const RooAbsArg* source) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void setTreeBranchStatus(TTree& t, Bool_t active) ;
  virtual void fillTreeBranch(TTree& t) ;

  mutable RooCatType _value ; // Current value
  TObjArray  _types ;         // Array of allowed values
  TIterator* _typeIter ;      //!

  ClassDef(RooAbsCategory,1) // Abstract index variable
};

#endif
