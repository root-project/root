/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddPdf.h,v 1.46 2007/07/12 20:30:28 wouter Exp $
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
#ifndef ROO_ADD_PDF
#define ROO_ADD_PDF

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"
#include "RooNormSetCache.h"
#include "RooObjCacheManager.h"
#include "RooNameReg.h"
#include "RooTrace.h"

#include <vector>
#include <list>
#include <utility>

class RooAddPdf : public RooAbsPdf {
public:

  RooAddPdf() : _projCacheMgr(this,10) { TRACE_CREATE }
  RooAddPdf(const char *name, const char *title=0);
  RooAddPdf(const char *name, const char *title,
            RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) ;
  RooAddPdf(const char *name, const char *title, const RooArgList& pdfList) ;
  RooAddPdf(const char *name, const char *title, const RooArgList& pdfList, const RooArgList& coefList, bool recursiveFraction=false) ;
  
  RooAddPdf(const RooAddPdf& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooAddPdf(*this,newname) ; }
  ~RooAddPdf() override { TRACE_DESTROY }

  bool checkObservables(const RooArgSet* nset) const override;

  /// Force RooRealIntegral to offer all observables for internal integration.
  bool forceAnalyticalInt(const RooAbsArg& /*dep*/) const override {
    return true ; 
  }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const override;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override;
  bool selfNormalized() const override {
    // P.d.f is self normalized
    return true ; 
  }

  ExtendMode extendMode() const override {
    // Return extended mode capabilities
    return ((_haveLastCoef&&!_recursive) || _allExtendable) ? MustBeExtended : CanNotBeExtended; 
  }
  /// Return expected number of events for extended likelihood calculation, which
  /// is the sum of all coefficients.
  Double_t expectedEvents(const RooArgSet* nset) const override;

  const RooArgList& pdfList() const { 
    // Return list of component p.d.fs
    return _pdfList ; 
  }
  const RooArgList& coefList() const { 
    // Return list of coefficients of component p.d.f.s
    return _coefList ; 
  }

  void fixCoefNormalization(const RooArgSet& refCoefNorm) ;  
  void fixCoefRange(const char* rangeName) ;
  
  const RooArgSet& getCoefNormalization() const { return _refCoefNorm ; }
  const char* getCoefRange() const { return _refCoefRangeName?RooNameReg::str(_refCoefRangeName):"" ; }

  void resetErrorCounters(Int_t resetValue=10) override;

  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const override;
  std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override;
  bool isBinnedDistribution(const RooArgSet& obs) const override;

  void printMetaArgs(std::ostream& os) const override;

  CacheMode canNodeBeCached() const override { return RooAbsArg::NotAdvised ; };
  void setCacheAndTrackHints(RooArgSet&) override;

protected:

  void selectNormalization(const RooArgSet* depSet=0, bool force=false) override;
  void selectNormalizationRange(const char* rangeName=0, bool force=false) override;

  mutable RooSetProxy _refCoefNorm ;   // Reference observable set for coefficient interpretation
  mutable TNamed* _refCoefRangeName = nullptr ;  // Reference range name for coefficient interpreation

  bool _projectCoefs = false;   // If true coefficients need to be projected for use in evaluate()
  std::vector<double> _coefCache; //! Transient cache with transformed values of coefficients


  class CacheElem : public RooAbsCacheElement {
  public:
    virtual ~CacheElem() {} ;

    RooArgList _suppNormList ; // Supplemental normalization list
    bool    _needSupNorm ; // Does the above list contain any non-unit entries?

    RooArgList _projList ; // Projection integrals to be multiplied with coefficients
    RooArgList _suppProjList ; // Projection integrals to be multiplied with coefficients for supplemental normalization terms
    RooArgList _refRangeProjList ; // Range integrals to be multiplied with coefficients (reference range)
    RooArgList _rangeProjList ; // Range integrals to be multiplied with coefficients (target range)

    virtual RooArgList containedArgs(Action) ;

  } ;
  mutable RooObjCacheManager _projCacheMgr ;  //! Manager of cache with coefficient projections and transformations
  CacheElem* getProjCache(const RooArgSet* nset, const RooArgSet* iset=0, const char* rangeName=0) const ;
  void updateCoefficients(CacheElem& cache, const RooArgSet* nset) const ;

  
  friend class RooAddGenContext ;
  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0, 
                               const RooArgSet* auxProto=0, bool verbose= false) const override;


  Double_t evaluate() const override {
      return getValV(nullptr);
  }
  double getValV(const RooArgSet* set=nullptr) const override ;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooBatchCompute::DataMap&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }


  mutable RooAICRegistry _codeReg; //! Registry of component analytical integration codes

  RooListProxy _pdfList ;   //  List of component PDFs
  RooListProxy _coefList ;  //  List of coefficients
  mutable RooArgList* _snormList{nullptr};  //!  List of supplemental normalization factors
  
  bool _haveLastCoef = false;  //  Flag indicating if last PDFs coefficient was supplied in the ctor
  bool _allExtendable = false; //  Flag indicating if all PDF components are extendable
  bool _recursive = false;     //  Flag indicating is fractions are treated recursively

  mutable Int_t _coefErrCount ; //! Coefficient error counter

  bool redirectServersHook(const RooAbsCollection&, bool, bool, bool) override {
    // If a server is redirected, the cached normalization set might not point
    // to the right observables anymore. We need to reset it.
    _copyOfLastNormSet.reset();
    return false;
  }

private:
  std::pair<const RooArgSet*, CacheElem*> getNormAndCache(const RooArgSet* nset) const;
  mutable RooFit::UniqueId<RooArgSet>::Value_t _idOfLastUsedNormSet = RooFit::UniqueId<RooArgSet>::nullval; ///<!
  mutable std::unique_ptr<const RooArgSet> _copyOfLastNormSet = nullptr; ///<!

  void finalizeConstruction();

  ClassDefOverride(RooAddPdf,4) // PDF representing a sum of PDFs
};

#endif
