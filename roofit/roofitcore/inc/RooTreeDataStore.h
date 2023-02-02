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
#ifndef ROO_TREE_DATA_STORE
#define ROO_TREE_DATA_STORE

#include "RooAbsDataStore.h"
#include "RunContext.h"

#include "ROOT/RStringView.hxx"

#include <vector>
#include <list>
#include <string>

class RooAbsArg ;
class RooArgList ;
class TTree ;
class RooFormulaVar ;
class RooArgSet ;


class RooTreeDataStore : public RooAbsDataStore {
public:

  RooTreeDataStore() ;
  RooTreeDataStore(TTree* t, const RooArgSet& vars, const char* wgtVarName=nullptr) ;

  // Empty ctor
  RooTreeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, const char* wgtVarName=nullptr) ;
  RooAbsDataStore* clone(const char* newname=nullptr) const override { return new RooTreeDataStore(*this,newname) ; }
  RooAbsDataStore* clone(const RooArgSet& vars, const char* newname=nullptr) const override { return new RooTreeDataStore(*this,vars,newname) ; }

  std::unique_ptr<RooAbsDataStore> reduce(RooStringView name, RooStringView title,
                          const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                          std::size_t nStart, std::size_t nStop) override;

  // Constructor from TTree
  RooTreeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, TTree& t, const char* selExpr=nullptr, const char* wgtVarName=nullptr) ;

  // Constructor from DataStore
  RooTreeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, const RooAbsDataStore& tds, const char* selExpr=nullptr, const char* wgtVarName=nullptr) ;

  RooTreeDataStore(RooStringView name, RooStringView title, RooAbsDataStore& tds,
                   const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                   Int_t nStart, Int_t nStop, const char* wgtVarName=nullptr) ;

  RooTreeDataStore(const RooTreeDataStore& other, const char* newname=nullptr) ;
  RooTreeDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname=nullptr) ;
  ~RooTreeDataStore() override ;


  // Write current row
  Int_t fill() override ;

  // Retrieve a row
  using RooAbsDataStore::get ;
  const RooArgSet* get(Int_t index) const override ;
  using RooAbsDataStore::weight ;
  double weight() const override ;
  double weightError(RooAbsData::ErrorType etype=RooAbsData::Poisson) const override ;
  void weightError(double& lo, double& hi, RooAbsData::ErrorType etype=RooAbsData::Poisson) const override ;
  bool isWeighted() const override { return (_wgtVar!=nullptr||_extWgtArray!=nullptr) ; }

  RooAbsData::RealSpans getBatches(std::size_t first, std::size_t len) const override {
    //TODO
    std::cerr << "This functionality is not yet implemented for tree data stores." << std::endl;
    throw std::logic_error("getBatches() not implemented in RooTreeDataStore.");
    (void)first; (void)len;
    return {};
  }
  RooSpan<const double> getWeightBatch(std::size_t first, std::size_t len) const override;

  // Change observable name
  bool changeObservableName(const char* from, const char* to) override ;

  // Add one column
  RooAbsArg* addColumn(RooAbsArg& var, bool adjustRange=true) override ;

  // Merge column-wise
  RooAbsDataStore* merge(const RooArgSet& allvars, std::list<RooAbsDataStore*> dstoreList) override ;

  // Add rows
  void append(RooAbsDataStore& other) override ;

  // General & bookkeeping methods
  double sumEntries() const override ;
  Int_t numEntries() const override ;
  void reset() override ;

  // Buffer redirection routines used in inside RooAbsOptTestStatistics
  void attachBuffers(const RooArgSet& extObs) override ;
  void resetBuffers() override ;
  void restoreAlternateBuffers() ;

  // Tree access
  TTree& tree() { return *_tree ; }
  const TTree* tree() const override { return _tree ; }

  // Forwarded from TTree
  Stat_t GetEntries() const;
  void Reset(Option_t* option=nullptr);
  Int_t Fill();
  Int_t GetEntry(Int_t entry = 0, Int_t getall = 0);

  void   Draw(Option_t* option = "") override ;

  // Constant term  optimizer interface
  void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=nullptr, bool skipZeroWeights=false) override ;
  const RooAbsArg* cacheOwner() override { return _cacheOwner ; }
  void setArgStatus(const RooArgSet& set, bool active) override ;
  void resetCache() override ;

  void loadValues(const TTree *t, const RooFormulaVar* select=nullptr, const char* rangeName=nullptr, Int_t nStart=0, Int_t nStop=2000000000)  ;
  void loadValues(const RooAbsDataStore *tds, const RooFormulaVar* select=nullptr, const char* rangeName=nullptr,
      std::size_t nStart=0, std::size_t nStop = std::numeric_limits<std::size_t>::max()) override;

  void checkInit() const override;

  void setExternalWeightArray(const double* arrayWgt, const double* arrayWgtErrLo,
      const double* arrayWgtErrHi, const double* arraySumW2) override {
    _extWgtArray = arrayWgt ;
    _extWgtErrLoArray = arrayWgtErrLo ;
    _extWgtErrHiArray = arrayWgtErrHi ;
    _extSumW2Array = arraySumW2 ;
  }

  const RooArgSet& row() { return _varsww ; }

 private:

  friend class RooVectorDataStore ;

  RooArgSet varsNoWeight(const RooArgSet& allVars, const char* wgtName=nullptr) ;
  RooRealVar* weightVar(const RooArgSet& allVars, const char* wgtName=nullptr) ;

  void initialize();
  void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) override ;

  // TTree Branch buffer size control
  void setBranchBufferSize(Int_t size) { _defTreeBufSize = size ; }
  Int_t getBranchBufferSize() const { return _defTreeBufSize ; }

  std::string makeTreeName() const;

  static Int_t _defTreeBufSize ;

  void createTree(RooStringView name, RooStringView title) ;
  TTree *_tree = nullptr;                 // TTree holding the data points
  TTree *_cacheTree = nullptr;            //! TTree holding the cached function values
  const RooAbsArg* _cacheOwner = nullptr; //! Object owning cache contents
  mutable bool _defCtor = false;        //! Was object constructed with default ctor?

  RooArgSet _varsww ;
  RooRealVar* _wgtVar = nullptr;     // Pointer to weight variable (if set)

  const double* _extWgtArray{nullptr};         ///<! External weight array
  const double* _extWgtErrLoArray{nullptr};    ///<! External weight array - low error
  const double* _extWgtErrHiArray{nullptr};    ///<! External weight array - high error
  const double* _extSumW2Array{nullptr};       ///<! External sum of weights array
  mutable std::unique_ptr<std::vector<double>> _weightBuffer; //! Buffer for weights in case a batch of values is requested.

  mutable double  _curWgt = 1.0;      ///< Weight of current event
  mutable double  _curWgtErrLo = 0.0; ///< Weight of current event
  mutable double  _curWgtErrHi = 0.0; ///< Weight of current event
  mutable double  _curWgtErr = 0.0;   ///< Weight of current event

  RooArgSet _attachedBuffers ; ///<! Currently attached buffers (if different from _varsww)

  ClassDefOverride(RooTreeDataStore, 2) // TTree-based Data Storage class
};


#endif
