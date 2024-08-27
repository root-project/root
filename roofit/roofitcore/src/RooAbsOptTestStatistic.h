/*
 * Project: RooFit
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_ABS_OPT_TEST_STATISTIC
#define ROO_ABS_OPT_TEST_STATISTIC

#include "RooAbsTestStatistic.h"
#include "RooSetProxy.h"
#include "RooCategoryProxy.h"
#include "TString.h"

class RooArgSet ;
class RooAbsData ;
class RooAbsReal ;

class RooAbsOptTestStatistic : public RooAbsTestStatistic {
public:

  // Constructors, assignment etc
  RooAbsOptTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
                         const RooArgSet& projDeps,
                         RooAbsTestStatistic::Configuration const& cfg);
  RooAbsOptTestStatistic(const RooAbsOptTestStatistic& other, const char* name=nullptr);
  ~RooAbsOptTestStatistic() override;

  double combinedValue(RooAbsReal** gofArray, Int_t nVal) const override ;

  RooAbsReal& function() { return *_funcClone ; }
  const RooAbsReal& function() const { return *_funcClone ; }

  RooAbsData& data() ;
  const RooAbsData& data() const ;


  const char* cacheUniqueSuffix() const override;

  // Override this to be always true to force calculation of likelihood without parameters
  bool isDerived() const override { return true ; }

  void seal(const char* notice="") { _sealed = true ; _sealNotice = notice ; }
  bool isSealed() const { return _sealed ; }
  const char* sealNotice() const { return _sealNotice.Data() ; }

private:
  void setUpBinSampling();

protected:

  bool setDataSlave(RooAbsData& data, bool cloneData=true, bool ownNewDataAnyway=false) override ;
  void initSlave(RooAbsReal& real, RooAbsData& indata, const RooArgSet& projDeps, const char* rangeName,
       const char* addCoefRangeName)  ;

  friend class RooAbsReal ;
  friend class RooAbsTestStatistic ;

  virtual bool allowFunctionCache() { return true ;  }
  void constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt=true) override ;

  bool redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive) override ;
  void printCompactTreeHook(std::ostream& os, const char* indent="") override ;
  virtual RooArgSet requiredExtraObservables() const { return RooArgSet() ; }
  void optimizeCaching() ;
  void optimizeConstantTerms(bool,bool=true) ;
  void runRecalculateCache(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const override;

  RooArgSet*  _normSet = nullptr;           ///< Pointer to set with observables used for normalization
  RooArgSet*  _funcCloneSet = nullptr;      ///< Set owning all components of internal clone of input function
  RooAbsData* _dataClone = nullptr; ///< Pointer to internal clone if input data
  RooAbsReal* _funcClone = nullptr;   ///< Pointer to internal clone of input function
  RooArgSet*  _projDeps = nullptr;    ///< Set of projected observable
  bool      _ownData = false;    ///< Do we own the dataset
  bool      _sealed = false;      ///< Is test statistic sealed -- i.e. no access to data
  TString     _sealNotice ;  ///< User-defined notice shown when reading a sealed likelihood
  RooArgSet*  _funcObsSet = nullptr;  ///< List of observables in the pdf expression
  RooArgSet   _cachedNodes ; ///<! List of nodes that are cached as constant expressions
  bool _skipZeroWeights = false; ///<! Whether to skip entries with weight zero in the evaluation

  RooAbsReal* _origFunc = nullptr;  ///< Original function
  RooAbsData* _origData = nullptr;  ///< Original data
  bool      _optimized = false; ///<!
  double      _integrateBinsPrecision{-1.}; // Precision for finer sampling of bins.
};

#endif
