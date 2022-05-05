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
#ifndef ROO_ABS_STUDY
#define ROO_ABS_STUDY

#include "TNamed.h"
#include "RooLinkedList.h"
#include "RooArgSet.h"

class RooAbsPdf;
class RooDataSet ;
class RooAbsData ;
class RooFitResult ;
class RooPlot ;
class RooRealVar ;
class RooWorkspace ;
class RooStudyManager ;
class RooStudyPackage ;

class RooAbsStudy : public TNamed {
public:

  RooAbsStudy() :  _storeDetails(false), _summaryData(0), _detailData(0), _ownDetailData(true) {} ;
  RooAbsStudy(const char* name, const char* title) ;
  RooAbsStudy(const RooAbsStudy& other) ;
  virtual RooAbsStudy* clone(const char* newname="") const = 0 ;
  TObject* Clone(const char* newname="") const override { return clone(newname) ; }
  ~RooAbsStudy() override ;

  virtual bool attach(RooWorkspace& /*w*/) { return false ; } ;
  virtual bool initialize() { return false ; } ;
  virtual bool execute() { return false ; } ;
  virtual bool finalize() { return 0 ; } ;
  void storeDetailedOutput(bool flag) { _storeDetails = flag ; }

  RooDataSet* summaryData() { return _summaryData ; }
  RooLinkedList* detailedData() { return _detailData ; }

  void releaseDetailData() { _ownDetailData = false ; }

  virtual void dump() {} ;

 protected:

  friend class RooStudyManager ;
  friend class RooStudyPackage ;
  void registerSummaryOutput(const RooArgSet& allVars, const RooArgSet& varsWithError=RooArgSet(), const RooArgSet& varsWithAsymError=RooArgSet()) ;
  void storeSummaryOutput(const RooArgSet& vars) ;
  void storeDetailedOutput(TNamed& object) ;
  void aggregateSummaryOutput(TList* chunkList) ;

 private:

  bool _storeDetails ;
  RooDataSet* _summaryData ;     ///<!
  RooLinkedList*  _detailData ;  ///<!
  bool      _ownDetailData ;

  ClassDefOverride(RooAbsStudy,1) // Abstract base class for RooStudyManager modules
} ;


#endif

