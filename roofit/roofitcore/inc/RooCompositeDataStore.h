/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_COMPOSITE_DATA_STORE
#define ROO_COMPOSITE_DATA_STORE

#include "RooAbsDataStore.h"

#include <map>
#include <string>
#include <vector>
#include <list>

class RooAbsArg ;
class RooArgList ;
class RooFormulaVar ;
class RooArgSet ;
class RooCategory ;


class RooCompositeDataStore : public RooAbsDataStore {
public:

  RooCompositeDataStore() ;

  // Ctors from DataStore
  RooCompositeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, RooCategory& indexCat, std::map<std::string,RooAbsDataStore*> const& inputData) ;

  // Empty ctor
  RooAbsDataStore* clone(const char* newname=nullptr) const override { return new RooCompositeDataStore(*this,newname) ; }
  RooAbsDataStore* clone(const RooArgSet& vars, const char* newname=nullptr) const override { return new RooCompositeDataStore(*this,vars,newname) ; }

  RooAbsDataStore* reduce(RooStringView name, RooStringView title,
                          const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                          std::size_t nStart, std::size_t nStop) override;

  RooCompositeDataStore(const RooCompositeDataStore& other, const char* newname=nullptr) ;
  RooCompositeDataStore(const RooCompositeDataStore& other, const RooArgSet& vars, const char* newname=nullptr) ;

  ~RooCompositeDataStore() override ;

  void dump() override ;

  // Write current row
  Int_t fill() override ;

  double sumEntries() const override ;

  // Retrieve a row
  using RooAbsDataStore::get ;
  const RooArgSet* get(Int_t index) const override ;
  using RooAbsDataStore::weight ;
  double weight() const override ;
  double weightError(RooAbsData::ErrorType etype=RooAbsData::Poisson) const override ;
  void weightError(double& lo, double& hi, RooAbsData::ErrorType etype=RooAbsData::Poisson) const override ;
  bool isWeighted() const override ;

  // Change observable name
  bool changeObservableName(const char* from, const char* to) override ;

  // Add one column
  RooAbsArg* addColumn(RooAbsArg& var, bool adjustRange=true) override ;

  // Merge column-wise
  RooAbsDataStore* merge(const RooArgSet& allvars, std::list<RooAbsDataStore*> dstoreList) override ;

  RooCategory* index() { return _indexCat ; }

  // Add rows
  void append(RooAbsDataStore& other) override ;

  // General & bookkeeping methods
  Int_t numEntries() const override ;
  void reset() override ;

  // Buffer redirection routines used in inside RooAbsOptTestStatistics
  void attachBuffers(const RooArgSet& extObs) override ;
  void resetBuffers() override ;

  // Constant term  optimizer interface
  void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=nullptr, bool skipZeroWeights=false) override ;
  const RooAbsArg* cacheOwner() override { return 0 ; }
  void setArgStatus(const RooArgSet& set, bool active) override ;
  void resetCache() override ;

  void recalculateCache(const RooArgSet* /*proj*/, Int_t /*firstEvent*/, Int_t /*lastEvent*/, Int_t /*stepSize*/, bool /*skipZeroWeights*/) override ;
  bool hasFilledCache() const override ;

  void loadValues(const RooAbsDataStore *tds, const RooFormulaVar* select=nullptr, const char* rangeName=nullptr,
      std::size_t nStart=0, std::size_t nStop = std::numeric_limits<std::size_t>::max()) override;

  void forceCacheUpdate() override ;

  RooAbsData::RealSpans getBatches(std::size_t first, std::size_t len) const override {
    //TODO
    std::cerr << "This functionality is not yet implemented for composite data stores." << std::endl;
    throw std::logic_error("getBatches() not implemented for RooCompositeDataStore.");
    (void)first; (void)len;
    return {};
  }
  RooSpan<const double> getWeightBatch(std::size_t first, std::size_t len) const override;


 protected:

  void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) override ;

  std::map<Int_t,RooAbsDataStore*> _dataMap ;
  RooCategory* _indexCat = nullptr;
  mutable RooAbsDataStore* _curStore = nullptr; ///<! Datastore associated with current event
  mutable Int_t _curIndex = 0; ///<! Index associated with current event
  mutable std::unique_ptr<std::vector<double>> _weightBuffer; ///<! Buffer for weights in case a batch of values is requested.
  bool _ownComps = false; ///<!

  ClassDefOverride(RooCompositeDataStore,1) // Composite Data Storage class
};


#endif
