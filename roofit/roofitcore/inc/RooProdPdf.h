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

namespace RooFit {
namespace Detail {
class RooFixedProdPdf;
}
}

class RooProdPdf : public RooAbsPdf {
public:

  RooProdPdf() ;
  RooProdPdf(const char *name, const char *title,
       RooAbsPdf& pdf1, RooAbsPdf& pdf2, double cutOff=0.0) ;
  RooProdPdf(const char* name, const char* title, const RooArgList& pdfList, double cutOff=0.0) ;
  RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet, const RooLinkedList& cmdArgList) ;

  RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet,
           const RooCmdArg& arg1            , const RooCmdArg& arg2={},
             const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
             const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
             const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;

  RooProdPdf(const char* name, const char* title,
             const RooCmdArg& arg1,             const RooCmdArg& arg2={},
             const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
             const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
             const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;

  RooProdPdf(const RooProdPdf& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooProdPdf(*this,newname) ; }
  ~RooProdPdf() override ;

  bool forceAnalyticalInt(const RooAbsArg& dep) const override ;
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  bool selfNormalized() const override { return _selfNorm ; }

  ExtendMode extendMode() const override ;
  double expectedEvents(const RooArgSet* nset) const override ;
  std::unique_ptr<RooAbsReal> createExpectedEventsFunc(const RooArgSet* nset) const override;

  const RooArgList& pdfList() const { return _pdfList ; }

  void addPdfs(RooAbsCollection const& pdfs);
  void removePdfs(RooAbsCollection const& pdfs);

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void initGenerator(Int_t code) override ;
  void generateEvent(Int_t code) override;
  bool isDirectGenSafe(const RooAbsArg& arg) const override ;

  // Constraint management
  RooArgSet* getConstraints(const RooArgSet& observables, RooArgSet const& constrainedParams, RooArgSet &pdfParams) const override ;

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

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

  // The cache object. Internal, do not use.
  class CacheElem final : public RooAbsCacheElement {
  public:
    CacheElem() : _isRearranged(false) { }
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

private:

  std::unique_ptr<RooArgSet> fillNormSetForServer(RooArgSet const& normSet, RooAbsArg const& server) const;

  double evaluate() const override ;

  std::unique_ptr<RooAbsReal> makeCondPdfRatioCorr(RooAbsReal& term, const RooArgSet& termNset, const RooArgSet& termImpSet, const char* normRange, const char* refRange) const ;

  void getParametersHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/, bool stripDisconnected) const override ;

  void initializeFromCmdArgList(const RooArgSet& fullPdfSet, const RooLinkedList& l) ;

  struct Factorized {
     ~Factorized();

     RooArgSet *termNormDeps(int i) const { return static_cast<RooArgSet*>(norms.At(i)); }
     RooArgSet *termIntDeps(int i) const { return static_cast<RooArgSet*>(ints.At(i)); }
     RooArgSet *termImpDeps(int i) const { return static_cast<RooArgSet*>(imps.At(i)); }
     RooArgSet *termCrossDeps(int i) const { return static_cast<RooArgSet*>(cross.At(i)); }

     RooLinkedList terms;
     RooLinkedList norms;
     RooLinkedList imps;
     RooLinkedList ints;
     RooLinkedList cross;
  };

  void factorizeProduct(const RooArgSet& normSet, const RooArgSet& intSet, Factorized &factorized) const;
  std::string makeRGPPName(const char* pfx, const RooArgSet& term, const RooArgSet& iset, const RooArgSet& nset, const char* isetRangeName) const ;
  void groupProductTerms(std::list<std::vector<RooArgSet*>>& groupedTerms, RooArgSet& outerIntDeps, Factorized const &factorized) const;



  Int_t getPartIntList(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName=nullptr) const ;

  struct ProcessProductTermOutput {
    bool isOwned = false;
    RooAbsReal* x0 = nullptr;
    std::unique_ptr<RooAbsReal> x1;
    std::unique_ptr<RooAbsReal> x2;
  };

  ProcessProductTermOutput processProductTerm(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName,
                     const RooArgSet* term,const RooArgSet& termNSet, const RooArgSet& termISet,
                     bool forceWrap=false) const ;


  CacheMode canNodeBeCached() const override { return RooAbsArg::NotAdvised ; } ;
  void setCacheAndTrackHints(RooArgSet&) override ;

  std::unique_ptr<CacheElem> createCacheElem(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName=nullptr) const;

  mutable RooObjCacheManager _cacheMgr ; //! The cache manager

  CacheElem* getCacheElem(RooArgSet const* nset) const ;
  void rearrangeProduct(CacheElem&) const;
  std::unique_ptr<RooAbsReal> specializeIntegral(RooAbsReal& orig, const char* targetRangeName) const ;
  std::unique_ptr<RooAbsReal> specializeRatio(RooFormulaVar& input, const char* targetRangeName) const ;
  double calculate(const RooProdPdf::CacheElem& cache, bool verbose=false) const ;
  void doEvalImpl(RooAbsArg const* caller, const RooProdPdf::CacheElem &cache, RooFit::EvalContext &) const;


  friend class RooProdGenContext ;
  friend class RooFit::Detail::RooFixedProdPdf ;
  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                  const RooArgSet *auxProto=nullptr, bool verbose= false) const override ;


  mutable RooAICRegistry _genCode ; ///<! Registry of composite direct generator codes

  double _cutOff = 0.0;       ///<  Cutoff parameter for running product
  RooListProxy _pdfList ;  ///<  List of PDF components
  std::vector<std::unique_ptr<RooArgSet>> _pdfNSetList ; ///< List of PDF component normalization sets
  Int_t _extendedIndex = -1; ///<  Index of extended PDF (if any)

  void useDefaultGen(bool flag=true) { _useDefaultGen = flag ; }
  bool _useDefaultGen = false; ///< Use default or distributed event generator

  mutable TNamed* _refRangeName = nullptr; ///< Reference range name for interpretation of conditional products

  bool _selfNorm = true; ///< Is self-normalized
  RooArgSet _defNormSet ; ///< Default normalization set

private:



  ClassDefOverride(RooProdPdf,6) // PDF representing a product of PDFs
};

namespace RooFit {
namespace Detail {

/// A RooProdPdf with a fixed normalization set can be replaced by this class.
/// Its purpose is to provide the right client-server interface for the
/// evaluation of RooProdPdf cache elements that were created for a given
/// normalization set.
class RooFixedProdPdf : public RooAbsPdf {
public:
   RooFixedProdPdf(std::unique_ptr<RooProdPdf> &&prodPdf, RooArgSet const &normSet);
   RooFixedProdPdf(const RooFixedProdPdf &other, const char *name = nullptr);

   inline TObject *clone(const char *newname) const override { return new RooFixedProdPdf(*this, newname); }

   inline bool selfNormalized() const override { return true; }

   inline bool canComputeBatchWithCuda() const override { return true; }

   inline void doEval(RooFit::EvalContext &ctx) const override { _prodPdf->doEvalImpl(this, *_cache, ctx); }

   inline ExtendMode extendMode() const override { return _prodPdf->extendMode(); }
   inline double expectedEvents(const RooArgSet * /*nset*/) const override
   {
      return _prodPdf->expectedEvents(&_normSet);
   }
   inline std::unique_ptr<RooAbsReal> createExpectedEventsFunc(const RooArgSet * /*nset*/) const override
   {
      return _prodPdf->createExpectedEventsFunc(&_normSet);
   }

   // Analytical Integration handling
   inline bool forceAnalyticalInt(const RooAbsArg &dep) const override { return _prodPdf->forceAnalyticalInt(dep); }
   inline Int_t getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &analVars, const RooArgSet *normSet,
                                        const char *rangeName = nullptr) const override
   {
      return _prodPdf->getAnalyticalIntegralWN(allVars, analVars, normSet, rangeName);
   }
   inline Int_t
   getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &numVars, const char *rangeName = nullptr) const override
   {
      return _prodPdf->getAnalyticalIntegral(allVars, numVars, rangeName);
   }
   inline double analyticalIntegralWN(Int_t code, const RooArgSet *normSet, const char *rangeName) const override
   {
      return _prodPdf->analyticalIntegralWN(code, normSet, rangeName);
   }
   inline double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override
   {
      return _prodPdf->analyticalIntegral(code, rangeName);
   }

   RooProdPdf::CacheElem const &cache() const { return *_cache; }

private:
   void initialize();

   inline double evaluate() const override { return _prodPdf->calculate(*_cache); }

   RooArgSet _normSet;
   std::unique_ptr<RooProdPdf::CacheElem> _cache;
   RooSetProxy _servers;
   std::unique_ptr<RooProdPdf> _prodPdf;

   ClassDefOverride(RooFit::Detail::RooFixedProdPdf, 0);
};

} // namespace Detail
} // namespace RooFit

#endif
