/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSimultaneous.h,v 1.42 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SIMULTANEOUS
#define ROO_SIMULTANEOUS

//#include "THashList.h"
#include "TList.h"
#include "RooAbsPdf.h"
#include "RooCategoryProxy.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"
#include "RooAbsCacheElement.h"
#include "RooArgList.h"
#include <map>
#include <string>
class RooAbsCategoryLValue ;
class RooFitResult ;
class RooPlot ;
class RooAbsData ;
class RooLinkedList ;

class RooSimultaneous : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooSimultaneous() : _plotCoefNormRange(0), _partIntMgr(this,10) {}
  RooSimultaneous(const char *name, const char *title, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const char *name, const char *title, std::map<std::string,RooAbsPdf*> pdfMap, RooAbsCategoryLValue& inIndexCat) ;
  RooSimultaneous(const char *name, const char *title, const RooArgList& pdfList, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const RooSimultaneous& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooSimultaneous(*this,newname) ; }
  ~RooSimultaneous() override ;

  double evaluate() const override ;
  bool selfNormalized() const override { return true ; }
  bool addPdf(const RooAbsPdf& pdf, const char* catLabel) ;

  ExtendMode extendMode() const override ;

  double expectedEvents(const RooArgSet* nset) const override ;

  bool forceAnalyticalInt(const RooAbsArg&) const override { return true ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;

  using RooAbsPdf::plotOn ;
  RooPlot* plotOn(RooPlot* frame,
           const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),
           const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
           const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
           const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg(),
           const RooCmdArg& arg9=RooCmdArg(), const RooCmdArg& arg10=RooCmdArg()) const override {
    return RooAbsReal::plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10) ;
  }
  RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const override ;

  // Backward compatibility function
  virtual RooPlot *plotOn(RooPlot *frame, Option_t* drawOptions, double scaleFactor=1.0,
           ScaleType stype=Relative, const RooAbsData* projData=nullptr, const RooArgSet* projSet=nullptr,
           double precision=1e-3, bool shiftToZero=false, const RooArgSet* projDataSet=nullptr,
           double rangeLo=0.0, double rangeHi=0.0, RooCurve::WingMode wmode=RooCurve::Extended) const;

  RooAbsPdf* getPdf(const char* catName) const ;
  const RooAbsCategoryLValue& indexCat() const { return (RooAbsCategoryLValue&) _indexCat.arg() ; }


  RooDataSet* generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents) override ;

  virtual RooDataHist* fillDataHist(RooDataHist *hist, const RooArgSet* nset, double scaleFactor,
                bool correctForBinVolume=false, bool showProgress=false) const ;

  void wrapPdfsInBinSamplingPdfs(RooAbsData const &data, double precision);
  void wrapPdfsInBinSamplingPdfs(RooAbsData const &data,
                                 std::map<std::string, double> const& precisions,
                                 bool useCategoryNames=false);

protected:

  void initialize(RooAbsCategoryLValue& inIndexCat, std::map<std::string,RooAbsPdf*> pdfMap) ;

  void selectNormalization(const RooArgSet* depSet=nullptr, bool force=false) override ;
  void selectNormalizationRange(const char* rangeName=nullptr, bool force=false) override ;
  mutable RooSetProxy _plotCoefNormSet ;
  const TNamed* _plotCoefNormRange ;

  class CacheElem : public RooAbsCacheElement {
  public:
    ~CacheElem() override {} ;
    RooArgList containedArgs(Action) override { return RooArgList(_partIntList) ; }
    RooArgList _partIntList ;
  } ;
  mutable RooObjCacheManager _partIntMgr ; ///<! Component normalization manager


  friend class RooSimGenContext ;
  friend class RooSimSplitGenContext ;
  RooAbsGenContext* autoGenContext(const RooArgSet &vars, const RooDataSet* prototype=nullptr, const RooArgSet* auxProto=nullptr,
                  bool verbose=false, bool autoBinned=true, const char* binnedTag="") const override ;
  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                  const RooArgSet* auxProto=nullptr, bool verbose= false) const override ;

  RooCategoryProxy _indexCat ; ///< Index category
  TList    _pdfProxyList ;     ///< List of PDF proxies (named after applicable category state)
  Int_t    _numPdf ;           ///< Number of registered PDFs

  ClassDefOverride(RooSimultaneous,3)  // Simultaneous operator p.d.f, functions like C++  'switch()' on input p.d.fs operating on index category5A
};

#endif
