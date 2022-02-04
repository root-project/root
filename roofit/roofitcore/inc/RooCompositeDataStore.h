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
#include "RunContext.h"

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
  RooCompositeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, RooCategory& indexCat, std::map<std::string,RooAbsDataStore*> inputData) ;

  // Empty ctor
  RooAbsDataStore* clone(const char* newname=0) const override { return new RooCompositeDataStore(*this,newname) ; }
  RooAbsDataStore* clone(const RooArgSet& vars, const char* newname=0) const override { return new RooCompositeDataStore(*this,vars,newname) ; }

  RooCompositeDataStore(const RooCompositeDataStore& other, const char* newname=0) ;
  RooCompositeDataStore(const RooCompositeDataStore& other, const RooArgSet& vars, const char* newname=0) ;
  ~RooCompositeDataStore() override ;

  void dump() override ;

  // Write current row
  Int_t fill() override ;

  Double_t sumEntries() const override ;

  // Retrieve a row
  using RooAbsDataStore::get ;
  const RooArgSet* get(Int_t index) const override ;
  using RooAbsDataStore::weight ;
  Double_t weight() const override ;
  Double_t weightError(RooAbsData::ErrorType etype=RooAbsData::Poisson) const override ;
  void weightError(Double_t& lo, Double_t& hi, RooAbsData::ErrorType etype=RooAbsData::Poisson) const override ;
  Bool_t isWeighted() const override ;

  // Change observable name
  Bool_t changeObservableName(const char* from, const char* to) override ;

  // Add one or more columns
  RooAbsArg* addColumn(RooAbsArg& var, Bool_t adjustRange=kTRUE) override ;
  RooArgSet* addColumns(const RooArgList& varList) override ;

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
  void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=0, Bool_t skipZeroWeights=kFALSE) override ;
  const RooAbsArg* cacheOwner() override { return 0 ; }
  void setArgStatus(const RooArgSet& set, Bool_t active) override ;
  void resetCache() override ;

  void recalculateCache(const RooArgSet* /*proj*/, Int_t /*firstEvent*/, Int_t /*lastEvent*/, Int_t /*stepSize*/, Bool_t /*skipZeroWeights*/) override ;
  Bool_t hasFilledCache() const override ;

  void loadValues(const RooAbsDataStore *tds, const RooFormulaVar* select=0, const char* rangeName=0,
      std::size_t nStart=0, std::size_t nStop = std::numeric_limits<std::size_t>::max()) override;

  void forceCacheUpdate() override ;

  RooBatchCompute::RunContext getBatches(std::size_t first, std::size_t len) const override {
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
  RooCategory* _indexCat ;
  mutable RooAbsDataStore* _curStore ; ///<! Datastore associated with current event
  mutable Int_t _curIndex ; ///<! Index associated with current event
  mutable std::unique_ptr<std::vector<double>> _weightBuffer; ///<! Buffer for weights in case a batch of values is requested.
  Bool_t _ownComps ; ///<!

  ClassDefOverride(RooCompositeDataStore,1) // Composite Data Storage class
};


#endif
