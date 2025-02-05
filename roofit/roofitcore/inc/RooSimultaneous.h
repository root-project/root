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

#include <RooAICRegistry.h>
#include <RooAbsCacheElement.h>
#include <RooAbsPdf.h>
#include <RooArgList.h>
#include <RooCategoryProxy.h>
#include <RooGlobalFunc.h>
#include <RooObjCacheManager.h>
#include <RooRealProxy.h>
#include <RooSetProxy.h>

#include <TList.h>

#include <map>
#include <string>

class RooAbsCategoryLValue ;
class RooFitResult ;
class RooPlot ;
class RooAbsData ;
class RooLinkedList ;
class RooSuperCategory ;

class RooSimultaneous : public RooAbsPdf {
public:

  /// Internal struct used for initialization.
  struct InitializationOutput {

     ~InitializationOutput();

     void addPdf(const RooAbsPdf &pdf, std::string const &catLabel);

     std::vector<RooAbsPdf const *> finalPdfs;
     std::vector<std::string> finalCatLabels;
     RooAbsCategoryLValue *indexCat = nullptr;
     std::unique_ptr<RooSuperCategory> superIndex;
  };

  // Constructors, assignment etc
  inline RooSimultaneous() : _partIntMgr(this,10) {}
  RooSimultaneous(const char *name, const char *title, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const char *name, const char *title, std::map<std::string,RooAbsPdf*> pdfMap, RooAbsCategoryLValue& inIndexCat) ;
  RooSimultaneous(const char *name, const char *title, RooFit::Detail::FlatMap<std::string,RooAbsPdf*> const &pdfMap, RooAbsCategoryLValue& inIndexCat);
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
           const RooCmdArg& arg1            , const RooCmdArg& arg2={},
           const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
           const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
           const RooCmdArg& arg7={}, const RooCmdArg& arg8={},
           const RooCmdArg& arg9={}, const RooCmdArg& arg10={}) const override {
    return RooAbsReal::plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10) ;
  }
  RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const override ;

  RooAbsPdf* getPdf(RooStringView catName) const ;
  const RooAbsCategoryLValue& indexCat() const { return (RooAbsCategoryLValue&) _indexCat.arg() ; }


  RooFit::OwningPtr<RooDataSet> generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents) override ;

  virtual RooDataHist* fillDataHist(RooDataHist *hist, const RooArgSet* nset, double scaleFactor,
                bool correctForBinVolume=false, bool showProgress=false) const ;

  void wrapPdfsInBinSamplingPdfs(RooAbsData const &data, double precision);
  void wrapPdfsInBinSamplingPdfs(RooAbsData const &data,
                                 std::map<std::string, double> const& precisions,
                                 bool useCategoryNames=false);

  RooAbsGenContext* autoGenContext(const RooArgSet &vars, const RooDataSet* prototype=nullptr, const RooArgSet* auxProto=nullptr,
                  bool verbose=false, bool autoBinned=true, const char* binnedTag="") const override ;
  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                  const RooArgSet* auxProto=nullptr, bool verbose= false) const override ;

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

protected:

  void selectNormalization(const RooArgSet* depSet=nullptr, bool force=false) override ;
  void selectNormalizationRange(const char* rangeName=nullptr, bool force=false) override ;

  RooArgSet const& flattenedCatList() const;

  mutable RooSetProxy _plotCoefNormSet ;
  const TNamed* _plotCoefNormRange = nullptr;

  class CacheElem : public RooAbsCacheElement {
  public:
    RooArgList containedArgs(Action) override { return RooArgList(_partIntList) ; }
    RooArgList _partIntList ;
  } ;
  mutable RooObjCacheManager _partIntMgr ; ///<! Component normalization manager


  friend class RooSimGenContext ;
  friend class RooSimSplitGenContext ;

  RooCategoryProxy _indexCat ; ///< Index category
  TList    _pdfProxyList ;     ///< List of PDF proxies (named after applicable category state)
  Int_t    _numPdf = 0;        ///< Number of registered PDFs

private:

  /// Private internal constructor.
  RooSimultaneous(const char *name, const char *title, InitializationOutput && initInfo);

  static std::unique_ptr<RooSimultaneous::InitializationOutput>
  initialize(std::string const& name, RooAbsCategoryLValue &inIndexCat,
             std::map<std::string, RooAbsPdf *> const &pdfMap);

  mutable std::unique_ptr<RooArgSet> _indexCatSet ; ///<! Index category wrapped in a RooArgSet if needed internally

  ClassDefOverride(RooSimultaneous,3)  // Simultaneous operator p.d.f, functions like C++  'switch()' on input p.d.fs operating on index category5A
};

#endif
