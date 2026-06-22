/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsAnaConvPdf.h,v 1.8 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_ABS_ANA_CONV_PDF
#define ROO_ABS_ANA_CONV_PDF


#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooDataSet.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"
#include "RooAbsCacheElement.h"
#include "RooResolutionModel.h"

class RooRealVar ;
class RooConvGenContext ;

class RooAbsAnaConvPdf : public RooAbsPdf {
public:

  // Constructors, assignment etc
  RooAbsAnaConvPdf() ;
  RooAbsAnaConvPdf(const char *name, const char *title,
         const RooResolutionModel& model,
         RooRealVar& convVar) ;

  RooAbsAnaConvPdf(const RooAbsAnaConvPdf& other, const char* name=nullptr);
  ~RooAbsAnaConvPdf() override;

  Int_t declareBasis(const char* expression, const RooArgList& params) ;
  void printMultiline(std::ostream& stream, Int_t contents, bool verbose=false, TString indent= "") const override ;

  // Coefficient normalization access
  inline double getCoefNorm(Int_t coefIdx, const RooArgSet& nset, const char* rangeName) const {
    // Returns normalization integral for coefficient coefIdx for observables nset in range rangeNae
    return getCoefNorm(coefIdx,&nset,rangeName) ;
  }
  double getCoefNorm(Int_t coefIdx, const RooArgSet* nset=nullptr, const char* rangeName=nullptr) const {
       return getCoefNorm(coefIdx,nset,RooNameReg::ptr(rangeName));
  }

  // Analytical integration support
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;

  // Coefficient Analytical integration support
  virtual Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const ;
  virtual double coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=nullptr) const ;
  bool forceAnalyticalInt(const RooAbsArg& dep) const override ;

  virtual double coefficient(Int_t basisIndex) const = 0 ;
  virtual RooFit::OwningPtr<RooArgSet> coefVars(Int_t coefIdx) const ;

  bool isDirectGenSafe(const RooAbsArg& arg) const override ;

  void setCacheAndTrackHints(RooArgSet&) override ;

  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                       const RooArgSet* auxProto=nullptr, bool verbose= false) const override ;
  virtual bool changeModel(const RooResolutionModel& newModel) ;

  /// Retrieve the convolution variable.
  RooAbsRealLValue* convVar();
  /// Retrieve the convolution variable.
  const RooAbsRealLValue* convVar() const {
    return const_cast<RooAbsAnaConvPdf*>(this)->convVar();
  }

  /// Get the resolution model.
  /// Note that the resolution model is only a configuration object specifying
  /// the model to convolve the basis functions with; it is not a server of this
  /// pdf and is not part of its computation graph (see the class documentation).
  RooAbsReal const &getModel() const { return *_model; }

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

protected:
  double getCoefNorm(Int_t coefIdx, const RooArgSet* nset, const TNamed* rangeName) const ;

  bool _isCopy ;

  double evaluate() const override ;

  bool redirectServersHook(const RooAbsCollection &newServerList, bool mustReplaceAll, bool nameChange,
                           bool isRecursive) override;

  void ioStreamerPass2() override;

  void makeCoefVarList(RooArgList&) const ;

  friend class RooConvGenContext ;

  // The original resolution model is intentionally *not* a server of the
  // RooAbsAnaConvPdf: it is only used to build the convolutions (stored in
  // _convSet) and for some operations like generation. Keeping it as a server
  // would pollute the computation graph (and any RooWorkspace or JSON export)
  // with an object that is never evaluated. It is owned and kept in sync with
  // the rest of the graph via redirectServersHook(), analogous to how
  // RooResolutionModel handles its basis function.
  RooResolutionModel *_model = nullptr; ///< Original resolution model (not a server)
  bool _ownModel = false;               ///< Flag indicating ownership of _model
  RooRealProxy _convVar ; ///< Convolution variable

  RooArgSet* parseIntegrationRequest(const RooArgSet& intSet, Int_t& coefCode, RooArgSet* analVars=nullptr) const ;

  RooListProxy _convSet  ;  ///<  Set of (resModel (x) basisFunc) convolution objects
  RooArgList _basisList ;   ///<!  List of created basis functions


  class CacheElem : public RooAbsCacheElement {
  public:
    RooArgList containedArgs(Action) override {
      RooArgList l(_coefVarList) ;
      l.add(_normList) ;
      return l ;
    }

    RooArgList _coefVarList ;
    RooArgList _normList ;
  } ;
  mutable RooObjCacheManager _coefNormMgr ; ///<! Coefficient normalization manager

  mutable RooAICRegistry _codeReg ;         ///<! Registry of analytical integration codes

  ClassDefOverride(RooAbsAnaConvPdf, 4) // Abstract Composite Convoluted PDF
};

#endif
