/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdPdf.h,v 1.44 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_PROD_PDF
#define ROO_PROD_PDF

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooLinkedList.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"
#include "RooCmdArg.h"

#include <vector>
#include <list>
#include <string>

typedef RooArgList* pRooArgList ;
typedef RooLinkedList* pRooLinkedList ;

class RooProdPdf : public RooAbsPdf {
public:

  RooProdPdf() ;
  RooProdPdf(const char *name, const char *title,
       RooAbsPdf& pdf1, RooAbsPdf& pdf2, double cutOff=0.0) ;
  RooProdPdf(const char* name, const char* title, const RooArgList& pdfList, double cutOff=0.0) ;
  RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet, const RooLinkedList& cmdArgList) ;

  RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet,
           const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),
             const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
             const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
             const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) ;

  RooProdPdf(const char* name, const char* title,
             const RooCmdArg& arg1,             const RooCmdArg& arg2=RooCmdArg(),
             const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
             const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
             const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) ;

  RooProdPdf(const RooProdPdf& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooProdPdf(*this,newname) ; }
  ~RooProdPdf() override ;

  bool checkObservables(const RooArgSet* nset) const override ;

  bool forceAnalyticalInt(const RooAbsArg& dep) const override ;
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  bool selfNormalized() const override { return _selfNorm ; }

  ExtendMode extendMode() const override ;
  double expectedEvents(const RooArgSet* nset) const override ;

  const RooArgList& pdfList() const { return _pdfList ; }

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void initGenerator(Int_t code) override ;
  void generateEvent(Int_t code) override;
  bool isDirectGenSafe(const RooAbsArg& arg) const override ;

  // Constraint management
  RooArgSet* getConstraints(const RooArgSet& observables, RooArgSet& constrainedParams, bool stripDisconnected) const override ;

  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;
  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  bool isBinnedDistribution(const RooArgSet& obs) const override  ;

  void printMetaArgs(std::ostream& os) const override ;

  void selectNormalizationRange(const char* rangeName=nullptr, bool force=false) override ;
  void fixRefRange(const char* rangeName) ;

  void setSelfNormalized(bool flag) { _selfNorm = flag ; }
  void setDefNormSet(const RooArgSet& nset) { _defNormSet.removeAll() ; _defNormSet.addClone(nset) ; }


  bool redirectServersHook(const RooAbsCollection& /*newServerList*/, bool /*mustReplaceAll*/, bool /*nameChange*/, bool /*isRecursive*/) override ;

  RooArgSet* getConnectedParameters(const RooArgSet& observables) const ;

  RooArgSet* findPdfNSet(RooAbsPdf const& pdf) const ;

  void writeCacheToStream(std::ostream& os, RooArgSet const* nset) const;

  std::unique_ptr<RooArgSet> fillNormSetForServer(RooArgSet const& normSet, RooAbsArg const& server) const override;

private:

  double evaluate() const override ;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

  RooAbsReal* makeCondPdfRatioCorr(RooAbsReal& term, const RooArgSet& termNset, const RooArgSet& termImpSet, const char* normRange, const char* refRange) const ;

  void getParametersHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/, bool stripDisconnected) const override ;

  void initializeFromCmdArgList(const RooArgSet& fullPdfSet, const RooLinkedList& l) ;

  void factorizeProduct(const RooArgSet& normSet, const RooArgSet& intSet,
                        RooLinkedList& termList,   RooLinkedList& normList,
                        RooLinkedList& impDepList, RooLinkedList& crossDepList,
                        RooLinkedList& intList) const;
  std::string makeRGPPName(const char* pfx, const RooArgSet& term, const RooArgSet& iset, const RooArgSet& nset, const char* isetRangeName) const ;
  void groupProductTerms(std::list<std::vector<RooArgSet*>>& groupedTerms, RooArgSet& outerIntDeps,
                         const RooLinkedList& terms, const RooLinkedList& norms,
                         const RooLinkedList& imps, const RooLinkedList& ints, const RooLinkedList& cross) const ;



  Int_t getPartIntList(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName=nullptr) const ;

  std::vector<RooAbsReal*> processProductTerm(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName,
                     const RooArgSet* term,const RooArgSet& termNSet, const RooArgSet& termISet,
                     bool& isOwned, bool forceWrap=false) const ;


  CacheMode canNodeBeCached() const override { return RooAbsArg::NotAdvised ; } ;
  void setCacheAndTrackHints(RooArgSet&) override ;

  // The cache object
  class CacheElem final : public RooAbsCacheElement {
  public:
    CacheElem() : _isRearranged(false) { }
    ~CacheElem() override = default;
    // Payload
    RooArgList _partList ;
    RooArgList _numList ;
    RooArgList _denList ;
    RooArgList _ownedList ;
    std::vector<std::unique_ptr<RooArgSet>> _normList;
    bool _isRearranged ;
    std::unique_ptr<RooAbsReal> _rearrangedNum{};
    std::unique_ptr<RooAbsReal> _rearrangedDen{};
    // Cache management functions
    RooArgList containedArgs(Action) override ;
    void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) override ;
    void writeToStream(std::ostream& os) const ;
  } ;
  mutable RooObjCacheManager _cacheMgr ; //! The cache manager

  CacheElem* getCacheElem(RooArgSet const* nset) const ;
  void rearrangeProduct(CacheElem&) const;
  RooAbsReal* specializeIntegral(RooAbsReal& orig, const char* targetRangeName) const ;
  RooAbsReal* specializeRatio(RooFormulaVar& input, const char* targetRangeName) const ;
  double calculate(const RooProdPdf::CacheElem& cache, bool verbose=false) const ;


  friend class RooProdGenContext ;
  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                  const RooArgSet *auxProto=nullptr, bool verbose= false) const override ;


  mutable RooAICRegistry _genCode ; ///<! Registry of composite direct generator codes

  double _cutOff ;       ///<  Cutoff parameter for running product
  RooListProxy _pdfList ;  ///<  List of PDF components
  std::vector<std::unique_ptr<RooArgSet>> _pdfNSetList ; ///< List of PDF component normalization sets
  Int_t _extendedIndex ;   ///<  Index of extended PDF (if any)

  void useDefaultGen(bool flag=true) { _useDefaultGen = flag ; }
  bool _useDefaultGen ; ///< Use default or distributed event generator

  mutable TNamed* _refRangeName ; ///< Reference range name for interpretation of conditional products

  bool _selfNorm ; ///< Is self-normalized
  RooArgSet _defNormSet ; ///< Default normalization set

private:

  ClassDefOverride(RooProdPdf,6) // PDF representing a product of PDFs
};


#endif
