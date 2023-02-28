/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
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
#ifndef ROO_MULTI_VAR_GAUSSIAN
#define ROO_MULTI_VAR_GAUSSIAN

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "TMatrixDSym.h"
#include "TMatrixD.h"
#include "TVectorD.h"

class RooRealVar;
class RooFitResult ;

#include <map>
#include <vector>

class RooMultiVarGaussian : public RooAbsPdf {
public:

  RooMultiVarGaussian() {} ;
  RooMultiVarGaussian(const char *name, const char *title, const RooArgList& xvec, const RooArgList& mu, const TMatrixDSym& covMatrix) ;
  RooMultiVarGaussian(const char *name, const char *title, const RooArgList& xvec, const RooFitResult& fr, bool reduceToConditional=true) ;
  RooMultiVarGaussian(const char *name, const char *title, const RooArgList& xvec, const TVectorD& mu, const TMatrixDSym& covMatrix) ;
  RooMultiVarGaussian(const char *name, const char *title, const RooArgList& xvec,const TMatrixDSym& covMatrix) ;
  void setAnaIntZ(double z) { _z = z ; }

  RooMultiVarGaussian(const RooMultiVarGaussian& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooMultiVarGaussian(*this,newname); }
  inline ~RooMultiVarGaussian() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void initGenerator(Int_t code) override ;
  void generateEvent(Int_t code) override;

  const TMatrixDSym& covarianceMatrix() const { return _cov ; }
  const RooArgList& xVec() const { return _x;}
  const RooArgList& muVec() const { return _mu; }

  class AnaIntData {
  public:
    TMatrixD    S22bar ;
    double    S22det ;
    std::vector<int> pmap ;
    Int_t       nint ;
  } ;

  class GenData {
  public:
    TMatrixD    UT ;
    std::vector<int> omap ;
    std::vector<int> pmap ;
    TVectorD    mu1 ;
    TVectorD    mu2 ;
    TMatrixD    S12S22I ;
  } ;

  class BitBlock {
  public:
    BitBlock() : b0(0), b1(0), b2(0), b3(0) {} ;

    void setBit(Int_t ibit) ;
    bool getBit(Int_t ibit) ;
    bool operator==(const BitBlock& other) ;

    Int_t b0 ;
    Int_t b1 ;
    Int_t b2 ;
    Int_t b3 ;
  } ;

  static void blockDecompose(const TMatrixD& input, const std::vector<int>& map1, const std::vector<int>& map2, TMatrixDSym& S11, TMatrixD& S12, TMatrixD& S21, TMatrixDSym& S22) ;

protected:

  void decodeCode(Int_t code, std::vector<int>& map1, std::vector<int>& map2) const;
  AnaIntData& anaIntData(Int_t code) const ;
  GenData& genData(Int_t code) const ;

  mutable std::map<int,AnaIntData> _anaIntCache ; ///<!
  mutable std::map<int,GenData> _genCache ; ///<!

  mutable std::vector<BitBlock> _aicMap ; ///<!

  RooListProxy _x ;
  RooListProxy _mu ;
  TMatrixDSym _cov ;
  TMatrixDSym _covI ;
  double    _det ;
  double    _z ;

  void syncMuVec() const ;
  mutable TVectorD _muVec ; //! Do not persist

  double evaluate() const override ;

private:

  ClassDefOverride(RooMultiVarGaussian,1) // Multivariate Gaussian PDF with correlations
};

#endif
