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

#include "RooAbsCategory.h"
#include "RooCategoryProxy.h"
#include <map>
#include <string>

class TRegexp;
class RooMappedCategoryCache;

class RooMappedCategory : public RooAbsCategory {
public:
  static constexpr value_type NoCatIdx = std::numeric_limits<value_type>::min();
  // Constructors etc.

  inline RooMappedCategory() : _defCat(0), _mapcache(nullptr) { }
  RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defCatName="NotMapped", Int_t defCatIdx=NoCatIdx);
  RooMappedCategory(const RooMappedCategory& other, const char *name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooMappedCategory(*this,newname); }
  ~RooMappedCategory() override;

  // Mapping function
  bool map(const char* inKeyRegExp, const char* outKeyName, Int_t outKeyNum=NoCatIdx) ;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override ;
  void printMetaArgs(std::ostream& os) const override ;

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;


  class Entry {
  public:
    inline Entry() : _regexp(nullptr), _catIdx() {}
    virtual ~Entry();
    Entry(const char* exp, RooAbsCategory::value_type cat);
    Entry(const Entry& other);
    bool ok();
    bool match(const char* testPattern) const;
    Entry& operator=(const Entry& other);
    RooAbsCategory::value_type outCat() const { return _catIdx; }
    const TRegexp* regexp() const;

  protected:

    TString mangle(const char* exp) const ;

    TString _expr ;
    mutable TRegexp* _regexp{nullptr}; ///<!
    RooAbsCategory::value_type _catIdx;

    ClassDef(Entry, 2) // Map cat entry definition
  };

protected:

  value_type _defCat{NoCatIdx}; ///< Default (unmapped) output type
  RooCategoryProxy _inputCat ;  ///< Input category
  std::map<std::string,RooMappedCategory::Entry> _mapArray ;  ///< List of mapping rules
  mutable RooMappedCategoryCache* _mapcache; ///<! transient member: cache the mapping

  value_type evaluate() const override ;
  const RooMappedCategoryCache* getOrCreateCache() const;

  /// When the input category changes states, the cached state mappings are invalidated
  void recomputeShape() override;

  friend class RooMappedCategoryCache;

  ClassDefOverride(RooMappedCategory, 2) // Index variable, derived from another index using pattern-matching based mapping
};

#endif
