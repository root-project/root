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

#include "RooAbsArg.h"
#include "RooCatType.h"
#include "TIterator.h"
#include "RooSpan.h"

class TTree ;
class RooArgSet ;
class RooDataSet ;
class Roo1DTable ;
class RooVectorDataStore ;

class RooAbsCategory : public RooAbsArg {
public:
  /// The type used to denote a specific category state.
  using value_type = int;

  // Constructors, assignment etc.
  RooAbsCategory() : _byteValue(0), _treeVar(false) { };
  RooAbsCategory(const char *name, const char *title);
  RooAbsCategory(const RooAbsCategory& other, const char* name=0) ;
  virtual ~RooAbsCategory();
  
  // Value accessors
  virtual value_type getIndex() const ;
  /// Retrieve a batch of category values for events in the range [begin, begin+batchSize).
  virtual RooSpan<const value_type> getValBatch(std::size_t /*begin*/, std::size_t /*batchSize*/) const {
    throw std::logic_error("Batch values are not implemented for RooAbsCategory.");
  }
  virtual const char* getLabel() const ;
  Bool_t operator==(value_type index) const ;
  Bool_t operator!=(value_type index) {  return !operator==(index);}
  Bool_t operator==(const char* label) const ;
  Bool_t operator!=(const char* label) { return !operator==(label);}
  virtual Bool_t operator==(const RooAbsArg& other) const ;
  Bool_t         operator!=(const RooAbsArg& other) { return !operator==(other);}
  virtual Bool_t isIdentical(const RooAbsArg& other, Bool_t assumeSameType=kFALSE) const;
  
  Bool_t isValidIndex(value_type index) const ;
  Bool_t isValidLabel(const char* label) const ;  
  const RooCatType* lookupType(value_type index, Bool_t printError=kFALSE) const ;
  const RooCatType* lookupType(const char* label, Bool_t printError=kFALSE) const ;
  const RooCatType* lookupType(const RooCatType& type, Bool_t printError=kFALSE) const ;
  /// \deprecated Iterator over types. Use range-based for loops instead.
  TIterator*
  R__SUGGEST_ALTERNATIVE("Use begin(), end() or range-based for loops.")
  typeIterator() const {
    return new LegacyIterator(_types);
  }
  /// Return number of types defined (in range named rangeName if rangeName!=0)
  Int_t numTypes(const char* /*rangeName*/=0) const { 
    return _types.size();
  }
  Bool_t isSignType(Bool_t mustHaveZero=kFALSE) const ;

  Roo1DTable *createTable(const char *label) const ;

  // I/O streaming interface
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const ;

  virtual void printValue(std::ostream& os) const ;
  virtual void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  virtual Bool_t isIntegrationSafeLValue(const RooArgSet* /*set*/) const { 
    // Is this l-value object safe for use as integration observable
    return kTRUE ; 
  }

  RooAbsArg *createFundamental(const char* newname=0) const;

  std::vector<RooCatType*>::const_iterator begin() const {
    return _types.cbegin();
  }

  std::vector<RooCatType*>::const_iterator end() const {
    return _types.cend();
  }

  std::size_t size() const {
    return _types.size();
  }

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
  const RooCatType* defineType(const char* label, value_type index) ;
  const RooCatType* defineTypeUnchecked(const char* label, value_type index) ;
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
  // These need to be pointers, unfortunately, since other classes are holding pointers to the categories.
  // That's not safe in case of reallocations.
  std::vector<RooCatType*> _types; // Vector of allowed values.

  Bool_t _treeVar ;           //! do not persist

  class LegacyIterator : public TIterator {
    public:
      LegacyIterator(const std::vector<RooCatType*>& vec) : _vec(&vec), index(-1) { }
      const TCollection *GetCollection() const override {
        return nullptr;
      }
      TObject* Next() override {
        ++index;
        return this->operator*();
      }
      void Reset() override {
        index = -1;
      }
      TObject* operator*() const override {
        // Need to const_cast, unfortunately because TIterator interface is too permissive
        return 0 <= index && index < (int)_vec->size() ? const_cast<RooCatType*>((*_vec)[index]) : nullptr;
      }
      LegacyIterator& operator=(const LegacyIterator&) = default;
      TIterator& operator=(const TIterator& other) override {
        auto otherLeg = dynamic_cast<LegacyIterator*>(*other);
        if (otherLeg)
          return this->operator=(*otherLeg);

        throw std::logic_error("Cannot assign to category iterators from incompatible types.");
      }

    private:
      const std::vector<RooCatType*>* _vec;
      int index;
  };

  ClassDef(RooAbsCategory, 2) // Abstract discrete variable
};

#endif
