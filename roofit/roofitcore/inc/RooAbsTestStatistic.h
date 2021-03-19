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

#include <stdexcept>
#include <string>
#include <vector>

class RooArgSet ;
class RooAbsData ;
class RooAbsReal ;
class RooSimultaneous ;
class RooRealMPFE ;

class RooAbsTestStatistic ;
typedef RooAbsTestStatistic* pRooAbsTestStatistic ;
typedef RooAbsData* pRooAbsData ;
typedef RooRealMPFE* pRooRealMPFE ;

class RooAbsTestStatistic : public RooAbsReal {
    friend class RooRealMPFE;
public:

  struct Configuration {

    /// Stores the configuration parameters for RooAbsTestStatistic.

    template<class T>
    struct DefaultValue { const T val; };

    template<class T>
    DefaultValue<T> makeDefault(T const& value) { return {value}; }

    template<class T>
    class ConfParam {

    /// Manages a configuration parameter.
    /// It supports setting a default value, can determine if a value has been
    /// set by the user, and allows to change the default value after
    /// construction.

    private:

      friend struct Configuration;

      /// Constructs a configuration parameter where the value is set to a default value
      /// that can still be changed after construction.
      /// This constructor should only be used in the declaration of the configuration struct.
      ConfParam(DefaultValue<T> const& value) : _isSet{false}, _value{value.val} {}

    public:

      /// Constructs a configuration parameter where the value is set immediately.
      ConfParam(T const& value) : _isSet{true}, _value{value} {}

      /// Check if a given configuration parameter is set.
      bool isSet() const { return _isSet ; }

      /// Set a configuration value after construction time.
      /// A given parameter should only be set once, otherwise this
      /// hints to an error in the code and an exception is thrown,
      /// hence the runtime_error throw.
      void setValue(T const& value) {
        if(_isSet) {
          throw std::runtime_error("Can't set the value of a configuration parameter that has already been set.");
        }
        _isSet = true;
        _value = value;
      }

      ConfParam& operator=(T const& value) {
        setValue(value);
        return *this;
      }

      /// Get the configuration parameter value.
      T const& operator*() const { return _value ; }

    private:

      friend class  RooChi2Var;

      /// Change the default value after construction time.
      /// This is a hack to reproduce the situation before, where the default
      /// values were set in the constructor parameters of RooAbsTestStatistic
      /// and its derived classes and the defaults were not consistent over the
      /// dependency hierachy.
      void setDefaultValue(T const& value) {
        if(!_isSet) {
          _value = value;
        }
      }

      bool _isSet = false;
      T _value{};
    };

    ConfParam<const char*> rangeName = makeDefault((const char*)nullptr);
    ConfParam<const char*> addCoefRangeName = makeDefault((const char*)nullptr);
    ConfParam<int> nCPU = makeDefault(1);
    ConfParam<RooFit::MPSplit> interleave = makeDefault(RooFit::BulkPartition);
    ConfParam<bool> verbose = makeDefault(true);
    ConfParam<bool> splitCutRange = makeDefault(false);
    ConfParam<bool> cloneInputData = makeDefault(true);
    ConfParam<double> integrateOverBinsPrecision = makeDefault(-1.);
    ConfParam<bool> binnedL = makeDefault(false);

  };

  // Constructors, assignment etc
  RooAbsTestStatistic() ;
  RooAbsTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
                      const RooArgSet& projDeps, Configuration && cfg);
  RooAbsTestStatistic(const RooAbsTestStatistic& other, const char* name=0);
  virtual ~RooAbsTestStatistic();
  virtual RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
                                      const RooArgSet& projDeps, Configuration && cfg) = 0;

  virtual void constOptimizeTestStatistic(ConstOpCode opcode, Bool_t doAlsoTrackingOpt=kTRUE) ;

  virtual Double_t combinedValue(RooAbsReal** gofArray, Int_t nVal) const = 0 ;
  virtual Double_t globalNormalization() const { 
    // Default value of global normalization factor is 1.0
    return 1.0 ; 
  }
  
  Bool_t setData(RooAbsData& data, Bool_t cloneData=kTRUE) ;

  void enableOffsetting(Bool_t flag) ;
  Bool_t isOffsetting() const { return _doOffset ; }
  virtual Double_t offset() const { return _offset.Sum() ; }
  virtual Double_t offsetCarry() const { return _offset.Carry(); }

protected:

  virtual void printCompactTreeHook(std::ostream& os, const char* indent="") ;

  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;
  virtual Double_t evaluate() const ;

  virtual Double_t evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const = 0 ;
  virtual Double_t getCarry() const;

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
  
  RooSetProxy _paramSet ;          // Parameters of the test statistic (=parameters of the input function)

  enum GOFOpMode { SimMaster,MPMaster,Slave } ;
  GOFOpMode operMode() const { 
    // Return test statistic operation mode of this instance (SimMaster, MPMaster or Slave)
    return _gofOpMode ; 
  }

  // Original arguments
  RooAbsReal* _func ;              // Pointer to original input function
  RooAbsData* _data ;              // Pointer to original input dataset
  const RooArgSet* _projDeps ;     // Pointer to set with projected observables
  std::string _rangeName ;         // Name of range in which to calculate test statistic
  std::string _addCoefRangeName ;  // Name of reference to be used for RooAddPdf components
  Bool_t _splitRange ;             // Split rangeName in RooSimultaneous index labels if true
  Int_t _simCount ;                // Total number of component p.d.f.s in RooSimultaneous (if any)
  Bool_t _verbose ;                // Verbose messaging if true

  virtual Bool_t setDataSlave(RooAbsData& /*data*/, Bool_t /*cloneData*/=kTRUE, Bool_t /*ownNewDataAnyway*/=kFALSE) { return kTRUE ; }

  //private:  


  virtual Bool_t processEmptyDataSets() const { return kTRUE ; }

  Bool_t initialize() ;
  void initSimMode(RooSimultaneous* pdf, RooAbsData* data, const RooArgSet* projDeps, const char* rangeName, const char* addCoefRangeName) ;    
  void initMPMode(RooAbsReal* real, RooAbsData* data, const RooArgSet* projDeps, const char* rangeName, const char* addCoefRangeName) ;

  mutable Bool_t _init ;          //! Is object initialized  
  GOFOpMode   _gofOpMode ;        // Operation mode of test statistic instance 

  Int_t       _nEvents ;          // Total number of events in test statistic calculation
  Int_t       _setNum ;           // Partition number of this instance in parallel calculation mode
  Int_t       _numSets ;          // Total number of partitions in parallel calculation mode
  Int_t       _extSet ;           //! Number of designated set to calculated extended term

  // Simultaneous mode data
  Int_t          _nGof        ; // Number of sub-contexts 
  pRooAbsTestStatistic* _gofArray ; //! Array of sub-contexts representing part of the combined test statistic
  std::vector<RooFit::MPSplit> _gofSplitMode ; //! GOF MP Split mode specified by component (when Auto is active)
  
  // Parallel mode data
  Int_t          _nCPU ;      //  Number of processors to use in parallel calculation mode
  pRooRealMPFE*  _mpfeArray ; //! Array of parallel execution frond ends

  RooFit::MPSplit        _mpinterl ; // Use interleaving strategy rather than N-wise split for partioning of dataset for multiprocessor-split
  Bool_t         _doOffset ; // Apply interval value offset to control numeric precision?
  mutable ROOT::Math::KahanSum<double> _offset{0.0} ; //! Offset as KahanSum to avoid loss of precision
  mutable Double_t _evalCarry; //! carry of Kahan sum in evaluatePartition

  ClassDef(RooAbsTestStatistic,2) // Abstract base class for real-valued test statistics

};

#endif
