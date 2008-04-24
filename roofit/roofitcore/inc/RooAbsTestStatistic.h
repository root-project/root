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

#include "Riosfwd.h"
#include "RooAbsReal.h"
#include "RooSetProxy.h"
#include <string>

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
public:

  // Constructors, assignment etc
  RooAbsTestStatistic() ;
  RooAbsTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
		      const RooArgSet& projDeps, const char* rangeName=0, const char* addCoefRangeName=0, 
		      Int_t nCPU=1, Bool_t interleave=kFALSE, Bool_t verbose=kTRUE, Bool_t splitCutRange=kTRUE) ;
  RooAbsTestStatistic(const RooAbsTestStatistic& other, const char* name=0);
  virtual ~RooAbsTestStatistic();
  virtual RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
				      const RooArgSet& projDeps, const char* rangeName=0, const char* addCoefRangeName=0, 
				      Int_t nCPU=1, Bool_t interleave=kFALSE, Bool_t verbose=kTRUE, Bool_t splitCutRange=kFALSE) = 0 ;

  virtual void constOptimizeTestStatistic(ConstOpCode opcode) ;

  virtual Double_t combinedValue(RooAbsReal** gofArray, Int_t nVal) const = 0 ;
  virtual Double_t globalNormalization() const { return 1.0 ; }

protected:

  virtual void printCompactTreeHook(ostream& os, const char* indent="") ;

  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;
  virtual Double_t evaluate() const ;

  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent, Int_t stepSize) const = 0 ;

  void setMPSet(Int_t setNum, Int_t numSets) ; 
  void setSimCount(Int_t simCount) { _simCount = simCount ; }
  void setEventCount(Int_t nEvents) { _nEvents = nEvents ; }

  Int_t numSets() const { return _numSets ; }
  Int_t setNum() const { return _setNum ; }
  
  RooSetProxy _paramSet ;

  enum GOFOpMode { SimMaster,MPMaster,Slave } ;
  GOFOpMode operMode() const { return _gofOpMode ; }

  // Original arguments
  RooAbsReal* _func ;
  RooAbsData* _data ;
  const RooArgSet* _projDeps ;
  std::string _rangeName ; 
  std::string _addCoefRangeName ; 
  Bool_t _splitRange ;
  Int_t _simCount ;
  Bool_t _verbose ;

private:  

  Bool_t initialize() ;
  void initSimMode(RooSimultaneous* pdf, RooAbsData* data, const RooArgSet* projDeps, const char* rangeName, const char* addCoefRangeName) ;    
  void initMPMode(RooAbsReal* real, RooAbsData* data, const RooArgSet* projDeps, const char* rangeName, const char* addCoefRangeName) ;

  mutable Bool_t _init ; //! 
  GOFOpMode   _gofOpMode ;

  Int_t       _nEvents ;
  Int_t       _setNum ;
  Int_t       _numSets ;

  // Simultaneous mode data
  Int_t          _nGof        ; // Number of sub-contexts 
  pRooAbsTestStatistic* _gofArray ; //! Array of sub-contexts representing part of the total NLL

  // Parallel mode data
  Int_t          _nCPU ;
  pRooRealMPFE*  _mpfeArray ; //! Array of parallel execution frond ends

  Bool_t         _mpinterl ; // Use interleaving strategy rather than N-wise split for partioning of dataset for multiprocessor-split

  ClassDef(RooAbsTestStatistic,1) // Abstract real-valued variable
};

#endif
