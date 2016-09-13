/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMappedCategory.h,v 1.22 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_MAPPED_CATEGORY
#define ROO_MAPPED_CATEGORY

#include "TObjArray.h"
#include "RooAbsCategory.h"
#include "RooCategoryProxy.h"
#include "RooCatType.h"
#include "TRegexp.h"
#include <map>
#include <string>

class RooMappedCategoryCache;

class RooMappedCategory : public RooAbsCategory {
public:
  // Constructors etc.
  enum CatIdx { NoCatIdx=-99999 } ;
  inline RooMappedCategory() : _defCat(0), _mapcache(0) { }
  RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defCatName="NotMapped", Int_t defCatIdx=NoCatIdx);
  RooMappedCategory(const RooMappedCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooMappedCategory(*this,newname); }
  virtual ~RooMappedCategory();

  // Mapping function
  Bool_t map(const char* inKeyRegExp, const char* outKeyName, Int_t outKeyNum=NoCatIdx) ; 

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const ;
  void printMetaArgs(std::ostream& os) const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const ;


  class Entry {
  public:
    inline Entry() : _regexp(0), _cat() {} 
    virtual ~Entry() { delete _regexp ; } ;
    Entry(const char* exp, const RooCatType* cat) : _expr(exp), _regexp(new TRegexp(mangle(exp),kTRUE)), _cat(*cat) {} 
    Entry(const Entry& other) : _expr(other._expr), _regexp(new TRegexp(mangle(other._expr.Data()),kTRUE)), _cat(other._cat) {} 
    inline Bool_t ok() { return (_regexp->Status()==TRegexp::kOK) ; }
    Bool_t match(const char* testPattern) const { return (TString(testPattern).Index(*_regexp)>=0) ; }
    inline const RooCatType& outCat() const { return _cat ; }
    Entry& operator=(const Entry& other);
    
  protected:
  
    TString mangle(const char* exp) const ;  

    TString _expr ;
    TRegexp* _regexp ; //!
    RooCatType _cat ;

    ClassDef(Entry,1) // Map cat entry definition
  };

protected:
    
  RooCatType* _defCat ;         // Default (unmapped) output type
  RooCategoryProxy _inputCat ;  // Input category
  std::map<std::string,RooMappedCategory::Entry> _mapArray ;  // List of mapping rules
  mutable RooMappedCategoryCache* _mapcache; //! transient member: cache the mapping

  virtual RooCatType evaluate() const ; 
  const RooMappedCategoryCache* getOrCreateCache() const;

  friend class RooMappedCategoryCache;

  ClassDef(RooMappedCategory,1) // Index variable, derived from another index using pattern-matching based mapping
};

#endif
