/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooResolutionModel.h,v 1.26 2007/05/14 18:37:46 wouter Exp $
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
#ifndef ROO_RESOLUTION_MODEL
#define ROO_RESOLUTION_MODEL

#include "RooAbsPdf.h"
#include "RooTemplateProxy.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"

class RooAbsAnaConvPdf;

class RooResolutionModel : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooResolutionModel() : _basis(0) { }
  RooResolutionModel(const char *name, const char *title, RooAbsRealLValue& x) ;
  RooResolutionModel(const RooResolutionModel& other, const char* name=0);
  TObject* clone(const char* newname) const override = 0 ;
  ~RooResolutionModel() override;

  virtual RooAbsGenContext* modelGenContext(const RooAbsAnaConvPdf&, const RooArgSet&,
                                            const RooDataSet*, const RooArgSet*,
                                            Bool_t) const { return 0; }

  Double_t getValV(const RooArgSet* nset=0) const override ;

  // If used as regular PDF, it also has to be normalized. If this resolution
  // model is used in a convolution, return unnormalized value regardless of
  // specified normalization set.
  bool selfNormalized() const override { return isConvolved() ; }

  virtual RooResolutionModel* convolution(RooFormulaVar* basis, RooAbsArg* owner) const ;
  /// Return the convolution variable of the resolution model.
  RooAbsRealLValue& convVar() const {return *x;}
  const RooRealVar& basisConvVar() const ;

  inline Bool_t isBasisSupported(const char* name) const { return basisCode(name)?kTRUE:kFALSE ; }
  virtual Int_t basisCode(const char* name) const = 0 ;

  virtual void normLeafServerList(RooArgSet& list) const ;
  Double_t getNorm(const RooArgSet* nset=0) const override ;

  inline const RooFormulaVar& basis() const { return _basis?*_basis:*identity() ; }
  Bool_t isConvolved() const { return _basis ? true : false ; }

  void printMultiline(std::ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const override ;

  static RooFormulaVar* identity() ;

  virtual void changeBasis(RooFormulaVar* basis) ;

protected:

  friend class RooConvGenContext ;
  friend class RooAddModel ;
  RooTemplateProxy<RooAbsRealLValue> x;                   ///< Dependent/convolution variable

  Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) override ;

  friend class RooAbsAnaConvPdf ;

  Int_t _basisCode ;         ///< Identifier code for selected basis function
  RooFormulaVar* _basis ;    ///< Basis function convolved with this resolution model
  Bool_t _ownBasis ;         ///< Flag indicating ownership of _basis

  ClassDefOverride(RooResolutionModel, 2) // Abstract Resolution Model
};

#endif
