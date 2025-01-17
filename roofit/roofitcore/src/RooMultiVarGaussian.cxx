/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
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

/**
\file RooMultiVarGaussian.cxx
\class RooMultiVarGaussian
\ingroup Roofitcore

Multivariate Gaussian p.d.f. with correlations
**/

#include "Riostream.h"
#include <cmath>

#include "RooMultiVarGaussian.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"
#include "RooGlobalFunc.h"
#include "RooConstVar.h"
#include "TDecompChol.h"
#include "RooFitResult.h"

using std::string, std::list, std::map, std::vector;

ClassImp(RooMultiVarGaussian);

////////////////////////////////////////////////////////////////////////////////

RooMultiVarGaussian::RooMultiVarGaussian(const char *name, const char *title,
                const RooArgList& xvec, const RooArgList& mu, const TMatrixDBase& cov) :
  RooAbsPdf(name,title),
  _x("x","Observables",this,true,false),
  _mu("mu","Offset vector",this,true,false),
  _cov{cov.GetNrows()},
  _covI{cov.GetNrows()},
  _z(4)
{
 if(!cov.IsSymmetric()) {
      std::stringstream errorMsg;
      errorMsg << "RooMultiVarGaussian::RooMultiVarGaussian(" << GetName()
               << ") input covariance matrix is not symmetric!";
      coutE(InputArguments) << errorMsg.str() << std::endl;
      throw std::invalid_argument(errorMsg.str().c_str());
 }

 _cov.SetSub(0, cov);
 _covI.SetSub(0, cov);

 _x.add(xvec) ;

 _mu.add(mu) ;

 _det = _cov.Determinant() ;

 // Invert covariance matrix
 _covI.Invert() ;
}


////////////////////////////////////////////////////////////////////////////////

RooMultiVarGaussian::RooMultiVarGaussian(const char *name, const char *title, const RooArgList &xvec,
                                         const RooFitResult &fr, bool reduceToConditional)
   : RooAbsPdf(name, title),
     _x("x", "Observables", this, true, false),
     _mu("mu", "Offset vector", this, true, false),
     _cov(reduceToConditional ? fr.conditionalCovarianceMatrix(xvec) : fr.reducedCovarianceMatrix(xvec)),
     _covI(_cov),
     _det(_cov.Determinant()),
     _z(4)
{

  // Fill mu vector with constant RooRealVars
  list<string> munames ;
  const RooArgList& fpf = fr.floatParsFinal() ;
  for (std::size_t i=0 ; i<fpf.size() ; i++) {
    if (xvec.find(fpf.at(i)->GetName())) {
      std::unique_ptr<RooRealVar> parclone{static_cast<RooRealVar*>(fpf.at(i)->Clone(Form("%s_centralvalue",fpf.at(i)->GetName())))};
      parclone->setConstant(true) ;
      _mu.addOwned(std::move(parclone));
      munames.push_back(fpf.at(i)->GetName()) ;
    }
  }

  // Fill X vector in same order as mu vector
  for (list<string>::iterator iter=munames.begin() ; iter!=munames.end() ; ++iter) {
    RooRealVar* xvar = static_cast<RooRealVar*>(xvec.find(iter->c_str())) ;
    _x.add(*xvar) ;
  }

  // Invert covariance matrix
  _covI.Invert() ;

}

namespace {

RooArgList muFromVector(TVectorD const &mu)
{
   RooArgList out;
   for (int i = 0; i < mu.GetNrows(); i++) {
      out.add(RooFit::RooConst(mu(i)));
   }
   return out;
}

RooArgList muConstants(std::size_t n)
{
   RooArgList out;
   for (std::size_t i = 0; i < n; i++) {
      out.add(RooFit::RooConst(0));
   }
   return out;
}

} // namespace


////////////////////////////////////////////////////////////////////////////////

RooMultiVarGaussian::RooMultiVarGaussian(const char *name, const char *title, const RooArgList &xvec,
                                         const TVectorD &mu, const TMatrixDBase &cov)
   : RooMultiVarGaussian{name, title, xvec, muFromVector(mu), cov}
{
}

////////////////////////////////////////////////////////////////////////////////

RooMultiVarGaussian::RooMultiVarGaussian(const char *name, const char *title, const RooArgList &xvec,
                                         const TMatrixDBase &cov)
   : RooMultiVarGaussian{name, title, xvec, muConstants(xvec.size()), cov}
{
}


////////////////////////////////////////////////////////////////////////////////

RooMultiVarGaussian::RooMultiVarGaussian(const RooMultiVarGaussian& other, const char* name) :
  RooAbsPdf(other,name), _aicMap(other._aicMap), _x("x",this,other._x), _mu("mu",this,other._mu),
  _cov(other._cov), _covI(other._covI), _det(other._det), _z(other._z)
{
}



////////////////////////////////////////////////////////////////////////////////

void RooMultiVarGaussian::syncMuVec() const
{
  _muVec.ResizeTo(_mu.size()) ;
  for (std::size_t i=0 ; i<_mu.size() ; i++) {
    _muVec[i] = static_cast<RooAbsReal*>(_mu.at(i))->getVal() ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Represent observables as vector

double RooMultiVarGaussian::evaluate() const
{
  TVectorD x(_x.size()) ;
  for (std::size_t i=0 ; i<_x.size() ; i++) {
    x[i] = static_cast<RooAbsReal*>(_x.at(i))->getVal() ;
  }

  // Calculate return value
  syncMuVec() ;
  TVectorD x_min_mu = x - _muVec ;

  double alpha =  x_min_mu * (_covI * x_min_mu) ;
  return exp(-0.5*alpha) ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooMultiVarGaussian::getAnalyticalIntegral(RooArgSet& allVarsIn, RooArgSet& analVars, const char* rangeName) const
{
  RooArgSet allVars(allVarsIn) ;

  // If allVars contains x_i it cannot contain mu_i
  for (std::size_t i=0 ; i<_x.size() ; i++) {
    if (allVars.contains(*_x.at(i))) {
      allVars.remove(*_mu.at(i),true,true) ;
    }
  }


  // Analytical integral known over all observables
  if (allVars.size()==_x.size() && !rangeName) {
    analVars.add(allVars) ;
    return -1 ;
  }

  Int_t code(0) ;

  Int_t nx = _x.size() ;
  if (nx>127) {
    // Warn that analytical integration is only provided for the first 127 observables
    coutW(Integration) << "RooMultiVarGaussian::getAnalyticalIntegral(" << GetName() << ") WARNING: p.d.f. has " << _x.size()
             << " observables, analytical integration is only implemented for the first 127 observables" << std::endl ;
    nx=127 ;
  }

  // Advertise partial analytical integral over all observables for which is wide enough to
  // use asymptotic integral calculation
  BitBlock bits ;
  bool anyBits(false) ;
  syncMuVec() ;
  for (std::size_t i=0 ; i<_x.size() ; i++) {

    // Check if integration over observable #i is requested
    if (allVars.find(_x.at(i)->GetName())) {
      // Check if range is wider than Z sigma
      RooRealVar* xi = static_cast<RooRealVar*>(_x.at(i)) ;
      if (xi->getMin(rangeName)<_muVec(i)-_z*sqrt(_cov(i,i)) && xi->getMax(rangeName) > _muVec(i)+_z*sqrt(_cov(i,i))) {
   cxcoutD(Integration) << "RooMultiVarGaussian::getAnalyticalIntegral(" << GetName()
              << ") Advertising analytical integral over " << xi->GetName() << " as range is >" << _z << " sigma" << std::endl ;
   bits.setBit(i) ;
   anyBits = true ;
   analVars.add(*allVars.find(_x.at(i)->GetName())) ;
      } else {
   cxcoutD(Integration) << "RooMultiVarGaussian::getAnalyticalIntegral(" << GetName() << ") Range of " << xi->GetName() << " is <"
              << _z << " sigma, relying on numeric integral" << std::endl ;
      }
    }

    // Check if integration over parameter #i is requested
    if (allVars.find(_mu.at(i)->GetName())) {
      // Check if range is wider than Z sigma
      RooRealVar* pi = static_cast<RooRealVar*>(_mu.at(i)) ;
      if (pi->getMin(rangeName)<_muVec(i)-_z*sqrt(_cov(i,i)) && pi->getMax(rangeName) > _muVec(i)+_z*sqrt(_cov(i,i))) {
   cxcoutD(Integration) << "RooMultiVarGaussian::getAnalyticalIntegral(" << GetName()
              << ") Advertising analytical integral over " << pi->GetName() << " as range is >" << _z << " sigma" << std::endl ;
   bits.setBit(i) ;
   anyBits = true ;
   analVars.add(*allVars.find(_mu.at(i)->GetName())) ;
      } else {
   cxcoutD(Integration) << "RooMultiVarGaussian::getAnalyticalIntegral(" << GetName() << ") Range of " << pi->GetName() << " is <"
              << _z << " sigma, relying on numeric integral" << std::endl ;
      }
    }


  }

  // Full numeric integration over requested observables maps always to code zero
  if (!anyBits) {
    return 0 ;
  }

  // Map BitBlock into return code
  for (UInt_t i=0 ; i<_aicMap.size() ; i++) {
    if (_aicMap[i]==bits) {
      code = i+1 ;
    }
  }
  if (code==0) {
    _aicMap.push_back(bits) ;
    code = _aicMap.size() ;
  }

  return code ;
}



////////////////////////////////////////////////////////////////////////////////
/// Handle full integral here

double RooMultiVarGaussian::analyticalIntegral(Int_t code, const char* /*rangeName*/) const
{
  if (code==-1) {
    return pow(2*3.14159268,_x.size()/2.)*sqrt(std::abs(_det)) ;
  }

  // Handle partial integrals here

  // Retrieve |S22|, S22bar from cache
  AnaIntData& aid = anaIntData(code) ;

  // Fill position vector for non-integrated observables
  syncMuVec() ;
  TVectorD u(aid.pmap.size()) ;
  for (UInt_t i=0 ; i<aid.pmap.size() ; i++) {
    u(i) = (static_cast<RooAbsReal*>(_x.at(aid.pmap[i])))->getVal() - _muVec(aid.pmap[i]) ;
  }

  // Calculate partial integral
  double ret = pow(2*3.14159268,aid.nint/2.)/sqrt(std::abs(aid.S22det))*exp(-0.5*u*(aid.S22bar*u)) ;

  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Check if cache entry was previously created

RooMultiVarGaussian::AnaIntData& RooMultiVarGaussian::anaIntData(Int_t code) const
{
  map<int,AnaIntData>::iterator iter =  _anaIntCache.find(code) ;
  if (iter != _anaIntCache.end()) {
    return iter->second ;
  }

  // Calculate cache contents

  // Decode integration code
  vector<int> map1;
  vector<int> map2;
  decodeCode(code,map1,map2) ;

  // Rearrange observables so that all non-integrated observables
  // go first (preserving relative order) and all integrated observables
  // go last (preserving relative order)
  TMatrixDSym S11;
  TMatrixDSym S22;
  TMatrixD S12;
  TMatrixD S21;
  blockDecompose(_covI,map1,map2,S11,S12,S21,S22) ;

  // Begin calculation of partial integrals
  //                                          ___
  //      sqrt(2pi)^(#intObs)     (-0.5 * u1T S22 u1 )
  // I =  ------------------- * e
  //        sqrt(|det(S22)|)
  //                                                                        ___
  // Where S22 is the sub-matrix of covI for the integrated observables and S22
  // is the Schur complement of S22
  // ___                   -1
  // S22  = S11 - S12 * S22   * S21
  //
  // and u1 is the vector of non-integrated observables

  // Calculate Schur complement S22bar
  TMatrixD S22inv(S22) ;
  S22inv.Invert() ;
  TMatrixD S22bar = S11 - S12*S22inv*S21 ;

  // Create new cache entry
  AnaIntData& cacheData = _anaIntCache[code] ;
  cacheData.S22bar.ResizeTo(S22bar) ;
  cacheData.S22bar=S22bar ;
  cacheData.S22det= S22.Determinant() ;
  cacheData.pmap = map1  ;
  cacheData.nint = map2.size() ;

  return cacheData ;
}



////////////////////////////////////////////////////////////////////////////////
/// Special case: generate all observables

Int_t RooMultiVarGaussian::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (directVars.size()==_x.size()) {
    generateVars.add(directVars) ;
    return -1 ;
  }

  Int_t nx = _x.size() ;
  if (nx>127) {
    // Warn that analytical integration is only provided for the first 127 observables
    coutW(Integration) << "RooMultiVarGaussian::getGenerator(" << GetName() << ") WARNING: p.d.f. has " << _x.size()
             << " observables, partial internal generation is only implemented for the first 127 observables" << std::endl ;
    nx=127 ;
  }

  // Advertise partial generation over all permutations of observables
  Int_t code(0) ;
  BitBlock bits ;
  for (std::size_t i=0 ; i<_x.size() ; i++) {
    RooAbsArg* arg = directVars.find(_x.at(i)->GetName()) ;
    if (arg) {
      bits.setBit(i) ;
//       code |= (1<<i) ;
      generateVars.add(*arg) ;
    }
  }

  // Map BitBlock into return code
  for (UInt_t i=0 ; i<_aicMap.size() ; i++) {
    if (_aicMap[i]==bits) {
      code = i+1 ;
    }
  }
  if (code==0) {
    _aicMap.push_back(bits) ;
    code = _aicMap.size() ;
  }


  return code ;
}



////////////////////////////////////////////////////////////////////////////////
/// Clear the GenData cache as its content is not invariant under changes in
/// the mu vector.

void RooMultiVarGaussian::initGenerator(Int_t /*code*/)
{
  _genCache.clear() ;

}




////////////////////////////////////////////////////////////////////////////////
/// Retrieve generator config from cache

void RooMultiVarGaussian::generateEvent(Int_t code)
{
  GenData& gd = genData(code) ;
  TMatrixD& TU = gd.UT ;
  Int_t nobs = TU.GetNcols() ;
  vector<int>& omap = gd.omap ;

  while(true) {

    // Create unit Gaussian vector
    TVectorD xgen(nobs);
    for(Int_t k= 0; k <nobs; k++) {
      xgen(k)= RooRandom::gaussian();
    }

    // Apply transformation matrix
    xgen *= TU ;

    // Apply shift
    if (code == -1) {

      // Simple shift if we generate all observables
      xgen += gd.mu1 ;

    } else {

      // Non-generated observable dependent shift for partial generations

      // mubar  = mu1 + S12 S22Inv ( x2 - mu2)
      TVectorD mubar(gd.mu1) ;
      TVectorD x2(gd.pmap.size()) ;
      for (UInt_t i=0 ; i<gd.pmap.size() ; i++) {
   x2(i) = (static_cast<RooAbsReal*>(_x.at(gd.pmap[i])))->getVal() ;
      }
      mubar += gd.S12S22I * (x2 - gd.mu2) ;

      xgen += mubar ;

    }

    // Transfer values and check if values are in range
    bool ok(true) ;
    for (int i=0 ; i<nobs ; i++) {
      RooRealVar* xi = static_cast<RooRealVar*>(_x.at(omap[i])) ;
      if (xgen(i)<xi->getMin() || xgen(i)>xi->getMax()) {
   ok = false ;
   break ;
      } else {
   xi->setVal(xgen(i)) ;
      }
    }

    // If all values are in range, accept event and return
    // otherwise retry
    if (ok) {
      break ;
    }
  }

  return;
}



////////////////////////////////////////////////////////////////////////////////
/// WVE -- CHECK THAT GENDATA IS VALID GIVEN CURRENT VALUES OF _MU

RooMultiVarGaussian::GenData& RooMultiVarGaussian::genData(Int_t code) const
{
  // Check if cache entry was previously created
  map<int,GenData>::iterator iter =  _genCache.find(code) ;
  if (iter != _genCache.end()) {
    return iter->second ;
  }

  // Create new entry
  GenData& cacheData = _genCache[code] ;

  if (code==-1) {

    // Do eigen value decomposition
    TDecompChol tdc(_cov) ;
    tdc.Decompose() ;
    TMatrixD U = tdc.GetU() ;
    TMatrixD TU(TMatrixD::kTransposed,U) ;

    // Fill cache data
    cacheData.UT.ResizeTo(TU) ;
    cacheData.UT = TU ;
    cacheData.omap.resize(_x.size()) ;
    for (std::size_t i=0 ; i<_x.size() ; i++) {
      cacheData.omap[i] = i ;
    }
    syncMuVec() ;
    cacheData.mu1.ResizeTo(_muVec) ;
    cacheData.mu1 = _muVec ;

  } else {

    // Construct observables: map1 = generated, map2 = given
    vector<int> map1;
    vector<int> map2;
    decodeCode(code,map2,map1) ;

    // Do block decomposition of covariance matrix
    TMatrixDSym S11;
    TMatrixDSym S22;
    TMatrixD S12;
    TMatrixD S21;
    blockDecompose(_cov,map1,map2,S11,S12,S21,S22) ;

    // Constructed conditional matrix form
    //                                             -1
    // F(X1|X2) --> CovI --> S22bar = S11 - S12 S22  S21
    //                                             -1
    //          --> mu   --> mubar  = mu1 + S12 S22  ( x2 - mu2)

    // Do eigenvalue decomposition
    TMatrixD S22Inv(TMatrixD::kInverted,S22) ;
    TMatrixD S22bar =  S11 - S12 * (S22Inv * S21) ;

    // Do eigen value decomposition of S22bar
    TDecompChol tdc(S22bar) ;
    tdc.Decompose() ;
    TMatrixD U = tdc.GetU() ;
    TMatrixD TU(TMatrixD::kTransposed,U) ;

    // Split mu vector into mu1 and mu2
    TVectorD mu1(map1.size());
    TVectorD mu2(map2.size());
    syncMuVec() ;
    for (UInt_t i=0 ; i<map1.size() ; i++) {
      mu1(i) = _muVec(map1[i]) ;
    }
    for (UInt_t i=0 ; i<map2.size() ; i++) {
      mu2(i) = _muVec(map2[i]) ;
    }

    // Calculate rotation matrix for mu vector
    TMatrixD S12S22Inv = S12 * S22Inv ;

    // Fill cache data
    cacheData.UT.ResizeTo(TU) ;
    cacheData.UT = TU ;
    cacheData.omap = map1 ;
    cacheData.pmap = map2 ;
    cacheData.mu1.ResizeTo(mu1) ;
    cacheData.mu2.ResizeTo(mu2) ;
    cacheData.mu1 = mu1 ;
    cacheData.mu2 = mu2 ;
    cacheData.S12S22I.ResizeTo(S12S22Inv) ;
    cacheData.S12S22I = S12S22Inv ;

  }


  return cacheData ;
}




////////////////////////////////////////////////////////////////////////////////
/// Decode analytical integration/generation code into index map of integrated/generated (map2)
/// and non-integrated/generated observables (map1)

void RooMultiVarGaussian::decodeCode(Int_t code, vector<int>& map1, vector<int>& map2) const
{
  if (code<0 || code> (Int_t)_aicMap.size()) {
    std::cout << "RooMultiVarGaussian::decodeCode(" << GetName() << ") ERROR don't have bit pattern for code " << code << std::endl ;
    throw string("RooMultiVarGaussian::decodeCode() ERROR don't have bit pattern for code") ;
  }

  BitBlock b = _aicMap[code-1] ;
  map1.clear() ;
  map2.clear() ;
  for (std::size_t i=0 ; i<_x.size() ; i++) {
    if (b.getBit(i)) {
      map2.push_back(i) ;
    } else {
      map1.push_back(i) ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Block decomposition of covI according to given maps of observables

void RooMultiVarGaussian::blockDecompose(const TMatrixD& input, const vector<int>& map1, const vector<int>& map2, TMatrixDSym& S11, TMatrixD& S12, TMatrixD& S21, TMatrixDSym& S22)
{
  // Allocate and fill reordered covI matrix in 2x2 block structure

  S11.ResizeTo(map1.size(),map1.size()) ;
  S12.ResizeTo(map1.size(),map2.size()) ;
  S21.ResizeTo(map2.size(),map1.size()) ;
  S22.ResizeTo(map2.size(),map2.size()) ;

  for (UInt_t i=0 ; i<map1.size() ; i++) {
    for (UInt_t j=0 ; j<map1.size() ; j++)
      S11(i,j) = input(map1[i],map1[j]) ;
    for (UInt_t j=0 ; j<map2.size() ; j++)
      S12(i,j) = input(map1[i],map2[j]) ;
  }
  for (UInt_t i=0 ; i<map2.size() ; i++) {
    for (UInt_t j=0 ; j<map1.size() ; j++)
      S21(i,j) = input(map2[i],map1[j]) ;
    for (UInt_t j=0 ; j<map2.size() ; j++)
      S22(i,j) = input(map2[i],map2[j]) ;
  }

}


void RooMultiVarGaussian::BitBlock::setBit(Int_t ibit)
{
  if (ibit<32) { b0 |= (1<<ibit) ; return ; }
  if (ibit<64) { b1 |= (1<<(ibit-32)) ; return ; }
  if (ibit<96) { b2 |= (1<<(ibit-64)) ; return ; }
  if (ibit<128) { b3 |= (1<<(ibit-96)) ; return ; }
}

bool RooMultiVarGaussian::BitBlock::getBit(Int_t ibit)
{
  if (ibit<32) return (b0 & (1<<ibit)) ;
  if (ibit<64) return (b1 & (1<<(ibit-32))) ;
  if (ibit<96) return (b2 & (1<<(ibit-64))) ;
  if (ibit<128) return (b3 & (1<<(ibit-96))) ;
  return false ;
}

bool operator==(RooMultiVarGaussian::BitBlock const &lhs, RooMultiVarGaussian::BitBlock const &rhs)
{
   if (lhs.b0 != rhs.b0)
      return false;
   if (lhs.b1 != rhs.b1)
      return false;
   if (lhs.b2 != rhs.b2)
      return false;
   if (lhs.b3 != rhs.b3)
      return false;
   return true;
}
