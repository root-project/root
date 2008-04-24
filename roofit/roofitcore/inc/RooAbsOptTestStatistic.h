/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsOptGoodnessOfFit.h,v 1.15 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_OPT_TEST_STATISTIC
#define ROO_ABS_OPT_TEST_STATISTIC

#include "Riosfwd.h"
#include "RooAbsTestStatistic.h"
#include "RooSetProxy.h"

class RooArgSet ;
class RooAbsData ;
class RooAbsReal ;

class RooAbsOptTestStatistic : public RooAbsTestStatistic {
public:

  // Constructors, assignment etc
  RooAbsOptTestStatistic() ;
  RooAbsOptTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
			 const RooArgSet& projDeps, const char* rangeName=0, const char* addCoefRangeName=0, 
			 Int_t nCPU=1, Bool_t interleave=kFALSE, Bool_t verbose=kTRUE, Bool_t splitCutRange=kFALSE) ;
  RooAbsOptTestStatistic(const RooAbsOptTestStatistic& other, const char* name=0);
  virtual ~RooAbsOptTestStatistic();

  virtual Double_t combinedValue(RooAbsReal** gofArray, Int_t nVal) const ;

protected:

  friend class RooAbsReal ;

  void constOptimizeTestStatistic(ConstOpCode opcode) ;
  
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;
  virtual void printCompactTreeHook(ostream& os, const char* indent="") ;

  void optimizeCaching() ;
  void optimizeConstantTerms(Bool_t) ;

  RooArgSet*  _normSet ;
  RooArgSet*  _funcCloneSet ;
  RooAbsData* _dataClone ;
  RooAbsReal* _funcClone ;
  RooArgSet*  _projDeps ;

  ClassDef(RooAbsOptTestStatistic,1) // Abstract real-valued variable
};

#endif
