/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCategory.h,v 1.38 2007/05/11 09:11:30 verkerke Exp $
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

#include "Riosfwd.h"
#include "TObjArray.h"
#include "RooAbsArg.h"
#include "RooCatType.h"

class TTree ;
class RooArgSet ;
class RooDataSet ;
class Roo1DTable ;
class RooVectorDataStore ;

class RooAbsCategory : public RooAbsArg {
public:
  // Constructors, assignment etc.
  RooAbsCategory() { 
    // Default constructor
    _treeVar = kFALSE ; 
    _typeIter = _types.MakeIterator() ; 
  } ;
  RooAbsCategory(const char *name, const char *title);
  RooAbsCategory(const RooAbsCategory& other, const char* name=0) ;
  virtual ~RooAbsCategory();
  
  // Value accessors
  virtual Int_t getIndex() const ;
  virtual const char* getLabel() const ;
  Bool_t operator==(Int_t index) const ;
  Bool_t operator!=(Int_t index) {  return !operator==(index);}
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
  Int_t numTypes(const char* /*rangeName*/=0) const { 
    // Return number of types defined (in range named rangeName if rangeName!=0)
    return _types.GetEntries() ; 
  }
  Bool_t isSignType(Bool_t mustHaveZero=kFALSE) const ;

  Roo1DTable *createTable(const char *label) const ;

  // I/O streaming interface
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  virtual void printValue(ostream& os) const ;
  virtual void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  virtual Bool_t isIntegrationSafeLValue(const RooArgSet* /*set*/) const { 
    // Is this l-value object safe for use as integration observable
    return kTRUE ; 
  }

  RooAbsArg *createFundamental(const char* newname=0) const;

protected:

  // Function evaluation and error tracing
  RooCatType traceEval() const ;
  // coverity[PASS_BY_VALUE]
  virtual Bool_t traceEvalHook(RooCatType /*value*/) const { 
    // Hook function for trace evaluation (dummy)
    return kFALSE ;
  }
  virtual RooCatType evaluate() const = 0 ;

  // Type definition management
  const RooCatType* defineType(const char* label) ;
  const RooCatType* defineType(const char* label, Int_t index) ;
  const RooCatType* defineTypeUnchecked(const char* label, Int_t index) ;
  const RooCatType* getOrdinal(UInt_t n, const char* rangeName=0) const;
  void clearTypes() ;

  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(const RooCatType& value) const ;

  friend class RooVectorDataStore ;
  virtual void syncCache(const RooArgSet* set=0) ;
  virtual void copyCache(const RooAbsArg* source, Bool_t valueOnly=kFALSE, Bool_t setValueDirty=kTRUE) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void attachToVStore(RooVectorDataStore& vstore) ;
  virtual void setTreeBranchStatus(TTree& t, Bool_t active) ;
  virtual void fillTreeBranch(TTree& t) ;

  mutable UChar_t    _byteValue ; //! Transient cache for byte values from tree branches
  mutable RooCatType _value ; // Current value
  TObjArray  _types ;         // Array of allowed values
  TIterator* _typeIter ;      //!

  Bool_t _treeVar ;           //! do not persist

  ClassDef(RooAbsCategory,1) // Abstract discrete variable
};

#endif
