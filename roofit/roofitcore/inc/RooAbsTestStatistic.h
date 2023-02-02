/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsGoodnessOfFit.h,v 1.15 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_TEST_STATISTIC
#define ROO_ABS_TEST_STATISTIC

#include "RooAbsReal.h"
#include "RooSetProxy.h"
#include "RooRealProxy.h"
#include "TStopwatch.h"
#include "Math/Util.h"

#include <string>
#include <vector>

class RooArgSet ;
class RooAbsData ;
class RooAbsReal ;
class RooSimultaneous ;
class RooRealMPFE ;

class RooAbsTestStatistic ;
typedef RooAbsData* pRooAbsData ;
typedef RooRealMPFE* pRooRealMPFE ;

class RooAbsTestStatistic : public RooAbsReal {
    friend class RooRealMPFE;
public:

  struct Configuration {
    /// Stores the configuration parameters for RooAbsTestStatistic.
    std::string rangeName = "";
    std::string addCoefRangeName = "";
    int nCPU = 1;
    RooFit::MPSplit interleave = RooFit::BulkPartition;
    bool verbose = true;
    bool splitCutRange = false;
    bool cloneInputData = true;
    double integrateOverBinsPrecision = -1.;
    bool binnedL = false;
    bool takeGlobalObservablesFromData = false;
  };

  // Constructors, assignment etc
  RooAbsTestStatistic() {}
  RooAbsTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
                      const RooArgSet& projDeps, Configuration const& cfg);
  RooAbsTestStatistic(const RooAbsTestStatistic& other, const char* name=nullptr);
  ~RooAbsTestStatistic() override;
  virtual RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
                                      const RooArgSet& projDeps, Configuration const& cfg) = 0;

  void constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt=true) override ;

  virtual double combinedValue(RooAbsReal** gofArray, Int_t nVal) const = 0 ;
  virtual double globalNormalization() const {
    // Default value of global normalization factor is 1.0
    return 1.0 ;
  }

  bool setData(RooAbsData& data, bool cloneData=true) override ;

  void enableOffsetting(bool flag) override ;
  bool isOffsetting() const override { return _doOffset ; }
  double offset() const override { return _offset.Sum() ; }
  virtual double offsetCarry() const { return _offset.Carry(); }

  enum GOFOpMode { SimMaster,MPMaster,Slave } ;
  GOFOpMode operMode() const {
     // Return test statistic operation mode of this instance (SimMaster, MPMaster or Slave)
     return _gofOpMode ;
  }

protected:

  void printCompactTreeHook(std::ostream& os, const char* indent="") override ;

  bool redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive) override ;
  double evaluate() const override ;

  virtual double evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const = 0 ;
  virtual double getCarry() const;

  void setMPSet(Int_t setNum, Int_t numSets) ;
  void setSimCount(Int_t simCount) {
    // Store total number of components p.d.f. of a RooSimultaneous in this component test statistic
    _simCount = simCount ;
  }

  void setEventCount(Int_t nEvents) {
    // Store total number of events in this component test statistic
    _nEvents = nEvents ;
  }

  Int_t numSets() const {
    // Return total number of sets for parallel calculation
    return _numSets ;
  }
  Int_t setNum() const {
    // Return parallel calculation set number for this instance
    return _setNum ;
  }

  RooSetProxy _paramSet ;          ///< Parameters of the test statistic (=parameters of the input function)


  // Original arguments
  RooAbsReal* _func = nullptr;          ///< Pointer to original input function
  RooAbsData* _data = nullptr;          ///< Pointer to original input dataset
  const RooArgSet* _projDeps = nullptr; ///< Pointer to set with projected observables
  std::string _rangeName ;              ///< Name of range in which to calculate test statistic
  std::string _addCoefRangeName ;       ///< Name of reference to be used for RooAddPdf components
  bool _splitRange = false;           ///< Split rangeName in RooSimultaneous index labels if true
  Int_t _simCount = 1;                  ///< Total number of component p.d.f.s in RooSimultaneous (if any)
  bool _verbose = false;              ///< Verbose messaging if true

  virtual bool setDataSlave(RooAbsData& /*data*/, bool /*cloneData*/=true, bool /*ownNewDataAnyway*/=false) { return true ; }

  //private:


  virtual bool processEmptyDataSets() const { return true ; }

  bool initialize() ;
  void initSimMode(RooSimultaneous* pdf, RooAbsData* data, const RooArgSet* projDeps, std::string const& rangeName, std::string const& addCoefRangeName) ;
  void initMPMode(RooAbsReal* real, RooAbsData* data, const RooArgSet* projDeps, std::string const& rangeName, std::string const& addCoefRangeName) ;

  mutable bool _init = false;   ///<! Is object initialized
  GOFOpMode _gofOpMode = Slave;   ///< Operation mode of test statistic instance

  Int_t _nEvents = 0;             ///< Total number of events in test statistic calculation
  Int_t       _setNum = 0;        ///< Partition number of this instance in parallel calculation mode
  Int_t       _numSets = 1;       ///< Total number of partitions in parallel calculation mode
  Int_t       _extSet = 0;        ///<! Number of designated set to calculated extended term

  // Simultaneous mode data
  std::vector<std::unique_ptr<RooAbsTestStatistic>> _gofArray; ///<! Array of sub-contexts representing part of the combined test statistic

  // Parallel mode data
  Int_t          _nCPU = 1;            ///<  Number of processors to use in parallel calculation mode
  pRooRealMPFE*  _mpfeArray = nullptr; ///<! Array of parallel execution frond ends

  RooFit::MPSplit _mpinterl = RooFit::BulkPartition;  ///< Use interleaving strategy rather than N-wise split for partioning of dataset for multiprocessor-split
  bool         _doOffset = false;                   ///< Apply interval value offset to control numeric precision?
  const bool  _takeGlobalObservablesFromData = false; ///< If the global observable values are taken from data
  mutable ROOT::Math::KahanSum<double> _offset {0.0}; ///<! Offset as KahanSum to avoid loss of precision
  mutable double _evalCarry = 0.0;                  ///<! carry of Kahan sum in evaluatePartition

  ClassDefOverride(RooAbsTestStatistic,0) // Abstract base class for real-valued test statistics

};

#endif
