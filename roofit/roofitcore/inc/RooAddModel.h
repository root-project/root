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
#ifndef ROO_ADD_MODEL
#define ROO_ADD_MODEL

#include "RooResolutionModel.h"
#include "RooListProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"
#include "RooNormSetCache.h"
#include "RooObjCacheManager.h"

class RooAddModel : public RooResolutionModel {
public:

  RooAddModel() ;
  RooAddModel(const char *name, const char *title, const RooArgList& pdfList, const RooArgList& coefList, bool ownPdfList=false) ;
  RooAddModel(const RooAddModel& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooAddModel(*this,newname) ; }
  RooResolutionModel* convolution(RooFormulaVar* basis, RooAbsArg* owner) const override ;
  ~RooAddModel() override ;

  Double_t evaluate() const override ;
  bool checkObservables(const RooArgSet* nset) const override ;

  Int_t basisCode(const char* name) const override ;

  bool forceAnalyticalInt(const RooAbsArg& /*dep*/) const override {
    // Force RooRealIntegral to offer all observables for internal integration
    return true ;
  }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const override ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override ;

  /// Model is self normalized when used as p.d.f
  bool selfNormalized() const override {
    return _basisCode==0 ? true : false ;
  }

  /// Return extended mode capabilities
  ExtendMode extendMode() const override {
    return (_haveLastCoef || _allExtendable) ? MustBeExtended : CanNotBeExtended;
  }

  /// Return expected number of events for extended likelihood calculation, which
  /// is the sum of all coefficients.
  Double_t expectedEvents(const RooArgSet* nset) const override ;

  /// Return list of component p.d.fs
  const RooArgList& pdfList() const {
    return _pdfList ;
  }

  /// Return list of coefficients of component p.d.f.s
  const RooArgList& coefList() const {
    return _coefList ;
  }

  bool isDirectGenSafe(const RooAbsArg& arg) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;


  void fixCoefNormalization(const RooArgSet& refCoefNorm) ;
  void fixCoefRange(const char* rangeName) ;
  void resetErrorCounters(Int_t resetValue=10) override ;

  void printMetaArgs(std::ostream& os) const override ;

protected:

  friend class RooAddGenContext ;
  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0,
                                       const RooArgSet* auxProto=0, bool verbose= false) const override ;

  void selectNormalization(const RooArgSet* depSet=0, bool force=false) override ;
  void selectNormalizationRange(const char* rangeName=0, bool force=false) override ;

  mutable RooSetProxy _refCoefNorm ;   ///<! Reference observable set for coefficient interpretation
  mutable TNamed* _refCoefRangeName ;  ///<! Reference range name for coefficient interpretation

  bool _projectCoefs ;         ///< If true coefficients need to be projected for use in evaluate()
  mutable Double_t* _coefCache ; ///<! Transiet cache with transformed values of coefficients


  class CacheElem : public RooAbsCacheElement {
  public:
    ~CacheElem() override {} ;

    RooArgList _suppNormList ;     ///< Supplemental normalization list

    RooArgList _projList ;         ///< Projection integrals to be multiplied with coefficients
    RooArgList _suppProjList ;     ///< Projection integrals to be multiplied with coefficients for supplemental normalization terms
    RooArgList _refRangeProjList ; ///< Range integrals to be multiplied with coefficients (reference range)
    RooArgList _rangeProjList ;    ///< Range integrals to be multiplied with coefficients (target range)

    RooArgList containedArgs(Action) override ;

  } ;
  mutable RooObjCacheManager _projCacheMgr ;  ///<! Manager of cache with coefficient projections and transformations
  CacheElem* getProjCache(const RooArgSet* nset, const RooArgSet* iset=0, const char* rangeName=0) const ;
  void updateCoefficients(CacheElem& cache, const RooArgSet* nset) const ;

  typedef RooArgList* pRooArgList ;
  void getCompIntList(const RooArgSet* nset, const RooArgSet* iset, pRooArgList& compIntList, Int_t& code, const char* isetRangeName) const ;
  class IntCacheElem : public RooAbsCacheElement {
  public:
    ~IntCacheElem() override {} ;
    RooArgList _intList ; ///< List of component integrals
    RooArgList containedArgs(Action) override ;
  } ;

  mutable RooObjCacheManager _intCacheMgr ; ///<! Manager of cache with integrals

  mutable RooAICRegistry _codeReg ;  ///<! Registry of component analytical integration codes

  RooListProxy _pdfList ;   ///<  List of component PDFs
  RooListProxy _coefList ;  ///<  List of coefficients
  mutable RooArgList* _snormList{nullptr};  ///<!  List of supplemental normalization factors

  bool _haveLastCoef ;    ///<  Flag indicating if last PDFs coefficient was supplied in the ctor
  bool _allExtendable ;   ///<  Flag indicating if all PDF components are extendable

  mutable Int_t _coefErrCount ; ///<! Coefficient error counter

  mutable RooArgSet _ownedComps ; ///<! Owned components

private:

  ClassDefOverride(RooAddModel,2) // Resolution model representing a sum of resolution models
};

#endif
