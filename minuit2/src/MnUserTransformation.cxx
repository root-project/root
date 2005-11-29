// @(#)root/minuit2:$Name:  $:$Id: MnUserTransformation.cpp,v 1.11.2.4 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnUserCovariance.h"

#include <algorithm>

namespace ROOT {

   namespace Minuit2 {


class MnParStr {

public:

  MnParStr(const char* Name) : fName(Name) {}

  ~MnParStr() {}
  
  bool operator()(const MinuitParameter& par) const {
    return (strcmp(par.Name(), fName) == 0);
  }

private:
  const char* fName;
};

MnUserTransformation::MnUserTransformation(const std::vector<double>& par, const std::vector<double>& err) : fPrecision(MnMachinePrecision()), fParameters(std::vector<MinuitParameter>()), fExtOfInt(std::vector<unsigned int>()), fDoubleLimTrafo(SinParameterTransformation()),fUpperLimTrafo(SqrtUpParameterTransformation()), fLowerLimTrafo(SqrtLowParameterTransformation()), fCache(std::vector<double>()) {
  fParameters.reserve(par.size());
  fExtOfInt.reserve(par.size());
  fCache.reserve(par.size());
  char p[5];
  p[0] = 'p';
  p[4] = '\0';
  for(unsigned int i = 0; i < par.size(); i++) {
    std::sprintf(p+1,"%i",i);
    Add(p, par[i], err[i]);
  }
}

const std::vector<double>& MnUserTransformation::operator()(const MnAlgebraicVector& pstates) const {

  for(unsigned int i = 0; i < pstates.size(); i++) {
    if(fParameters[fExtOfInt[i]].HasLimits()) {
      fCache[fExtOfInt[i]] = Int2ext(i, pstates(i));
    } else {
      fCache[fExtOfInt[i]] = pstates(i);
    }
  }

  return fCache;
}

double MnUserTransformation::Int2ext(unsigned int i, double val) const {

  if(fParameters[fExtOfInt[i]].HasLimits()) {
    if(fParameters[fExtOfInt[i]].HasUpperLimit() && fParameters[fExtOfInt[i]].HasLowerLimit())
      return fDoubleLimTrafo.Int2ext(val, fParameters[fExtOfInt[i]].UpperLimit(), fParameters[fExtOfInt[i]].LowerLimit());
    else if(fParameters[fExtOfInt[i]].HasUpperLimit() && !fParameters[fExtOfInt[i]].HasLowerLimit())
      return fUpperLimTrafo.Int2ext(val, fParameters[fExtOfInt[i]].UpperLimit());
    else
      return fLowerLimTrafo.Int2ext(val, fParameters[fExtOfInt[i]].LowerLimit());
  }

  return val;
}

double MnUserTransformation::Int2extError(unsigned int i, double val, double err) const {
  //err = sigma Value == sqrt(cov(i,i))
  double dx = err;
  
  if(fParameters[fExtOfInt[i]].HasLimits()) {
    double ui = Int2ext(i, val);
    double du1 = Int2ext(i, val+dx) - ui;
    double du2 = Int2ext(i, val-dx) - ui;
    if(fParameters[fExtOfInt[i]].HasUpperLimit() && fParameters[fExtOfInt[i]].HasLowerLimit()) {
//       double al = fParameters[fExtOfInt[i]].Lower();
//       double ba = fParameters[fExtOfInt[i]].Upper() - al;
//       double du1 = al + 0.5*(sin(val + dx) + 1.)*ba - ui;
//       double du2 = al + 0.5*(sin(val - dx) + 1.)*ba - ui;
//       if(dx > 1.) du1 = ba;
      if(dx > 1.) du1 = fParameters[fExtOfInt[i]].UpperLimit() - fParameters[fExtOfInt[i]].LowerLimit();
      dx = 0.5*(fabs(du1) + fabs(du2));
    } else {
      dx = 0.5*(fabs(du1) + fabs(du2));
    }
  }

  return dx;
}

MnUserCovariance MnUserTransformation::Int2extCovariance(const MnAlgebraicVector& vec, const MnAlgebraicSymMatrix& cov) const {
  
  MnUserCovariance result(cov.Nrow());
  for(unsigned int i = 0; i < vec.size(); i++) {
    double dxdi = 1.;
    if(fParameters[fExtOfInt[i]].HasLimits()) {
//       dxdi = 0.5*fabs((fParameters[fExtOfInt[i]].Upper() - fParameters[fExtOfInt[i]].Lower())*cos(vec(i)));
      dxdi = DInt2Ext(i, vec(i));
    }
    for(unsigned int j = i; j < vec.size(); j++) {
      double dxdj = 1.;
      if(fParameters[fExtOfInt[j]].HasLimits()) {
// 	dxdj = 0.5*fabs((fParameters[fExtOfInt[j]].Upper() - fParameters[fExtOfInt[j]].Lower())*cos(vec(j)));
	dxdj = DInt2Ext(j, vec(j));
      }
      result(i,j) = dxdi*cov(i,j)*dxdj;
    }
//     double diag = Int2extError(i, vec(i), sqrt(cov(i,i)));
//     result(i,i) = diag*diag;
  }
  
  return result;
}

double MnUserTransformation::Ext2int(unsigned int i, double val) const {

  if(fParameters[i].HasLimits()) {
    if(fParameters[i].HasUpperLimit() && fParameters[i].HasLowerLimit())
      return fDoubleLimTrafo.Ext2int(val, fParameters[i].UpperLimit(), fParameters[i].LowerLimit(), Precision());
    else if(fParameters[i].HasUpperLimit() && !fParameters[i].HasLowerLimit())
      return fUpperLimTrafo.Ext2int(val, fParameters[i].UpperLimit(), Precision());
    else 
      return fLowerLimTrafo.Ext2int(val, fParameters[i].LowerLimit(), Precision());
  }
  
  return val;
}

double MnUserTransformation::DInt2Ext(unsigned int i, double val) const {
  double dd = 1.;
  if(fParameters[fExtOfInt[i]].HasLimits()) {
    if(fParameters[fExtOfInt[i]].HasUpperLimit() && fParameters[fExtOfInt[i]].HasLowerLimit())  
//       dd = 0.5*fabs((fParameters[fExtOfInt[i]].Upper() - fParameters[fExtOfInt[i]].Lower())*cos(vec(i)));
      dd = fDoubleLimTrafo.DInt2Ext(val, fParameters[fExtOfInt[i]].UpperLimit(), fParameters[fExtOfInt[i]].LowerLimit());
    else if(fParameters[fExtOfInt[i]].HasUpperLimit() && !fParameters[fExtOfInt[i]].HasLowerLimit())
      dd = fUpperLimTrafo.DInt2Ext(val, fParameters[fExtOfInt[i]].UpperLimit());
    else 
      dd = fLowerLimTrafo.DInt2Ext(val, fParameters[fExtOfInt[i]].LowerLimit());
  }

  return dd;
}

/*
double MnUserTransformation::dExt2Int(unsigned int, double) const {
  double dd = 1.;

  if(fParameters[fExtOfInt[i]].HasLimits()) {
    if(fParameters[fExtOfInt[i]].HasUpperLimit() && fParameters[fExtOfInt[i]].HasLowerLimit())  
//       dd = 0.5*fabs((fParameters[fExtOfInt[i]].Upper() - fParameters[fExtOfInt[i]].Lower())*cos(vec(i)));
      dd = fDoubleLimTrafo.dExt2Int(val, fParameters[fExtOfInt[i]].UpperLimit(), fParameters[fExtOfInt[i]].LowerLimit());
    else if(fParameters[fExtOfInt[i]].HasUpperLimit() && !fParameters[fExtOfInt[i]].HasLowerLimit())
      dd = fUpperLimTrafo.dExt2Int(val, fParameters[fExtOfInt[i]].UpperLimit());
    else 
      dd = fLowerLimTrafo.dExtInt(val, fParameters[fExtOfInt[i]].LowerLimit());
  }

  return dd;
}
*/

unsigned int MnUserTransformation::IntOfExt(unsigned int ext) const {
  assert(ext < fParameters.size());
  assert(!fParameters[ext].IsFixed());
  assert(!fParameters[ext].IsConst());
  std::vector<unsigned int>::const_iterator iind = std::find(fExtOfInt.begin(), fExtOfInt.end(), ext);
  assert(iind != fExtOfInt.end());

  return (iind - fExtOfInt.begin());  
}

std::vector<double> MnUserTransformation::Params() const {
  std::vector<double> result; result.reserve(fParameters.size());
  for(std::vector<MinuitParameter>::const_iterator ipar = Parameters().begin();
      ipar != Parameters().end(); ipar++)
    result.push_back((*ipar).Value());

  return result;
}

std::vector<double> MnUserTransformation::Errors() const {
  std::vector<double> result; result.reserve(fParameters.size());
  for(std::vector<MinuitParameter>::const_iterator ipar = Parameters().begin();
      ipar != Parameters().end(); ipar++)
    result.push_back((*ipar).Error());
  
  return result;
}

const MinuitParameter& MnUserTransformation::Parameter(unsigned int n) const {
  assert(n < fParameters.size()); 
  return fParameters[n];
}

bool MnUserTransformation::Add(const char* Name, double val, double err) {
  if (std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name)) != fParameters.end() ) 
    return false; 
  fExtOfInt.push_back(fParameters.size());
  fCache.push_back(val);
  fParameters.push_back(MinuitParameter(fParameters.size(), Name, val, err));
  return true;
}

bool MnUserTransformation::Add(const char* Name, double val, double err, double low, double up) {
  if (std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name)) != fParameters.end() ) 
    return false; 
  fExtOfInt.push_back(fParameters.size());
  fCache.push_back(val);
  fParameters.push_back(MinuitParameter(fParameters.size(), Name, val, err, low, up));
  return true;
}

bool MnUserTransformation::Add(const char* Name, double val) {
  if (std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name)) != fParameters.end() ) 
    return false; 
  fCache.push_back(val);
  fParameters.push_back(MinuitParameter(fParameters.size(), Name, val));
  return true;
}

void MnUserTransformation::Fix(unsigned int n) {
  assert(n < fParameters.size()); 
  std::vector<unsigned int>::iterator iind = std::find(fExtOfInt.begin(), fExtOfInt.end(), n);
  assert(iind != fExtOfInt.end());
  fExtOfInt.erase(iind, iind+1);
  fParameters[n].Fix();
}

void MnUserTransformation::Release(unsigned int n) {
  assert(n < fParameters.size()); 
  std::vector<unsigned int>::const_iterator iind = std::find(fExtOfInt.begin(), fExtOfInt.end(), n);
  assert(iind == fExtOfInt.end());
  fExtOfInt.push_back(n);
  std::sort(fExtOfInt.begin(), fExtOfInt.end());
  fParameters[n].Release();
}

void MnUserTransformation::SetValue(unsigned int n, double val) {
  assert(n < fParameters.size()); 
  fParameters[n].SetValue(val);
  fCache[n] = val;
}

void MnUserTransformation::SetError(unsigned int n, double err) {
  assert(n < fParameters.size()); 
  fParameters[n].SetError(err);
}

void MnUserTransformation::SetLimits(unsigned int n, double low, double up) {
  assert(n < fParameters.size());
  assert(low != up);
  fParameters[n].SetLimits(low, up);
}

void MnUserTransformation::SetUpperLimit(unsigned int n, double up) {
  assert(n < fParameters.size()); 
  fParameters[n].SetUpperLimit(up);
}

void MnUserTransformation::SetLowerLimit(unsigned int n, double lo) {
  assert(n < fParameters.size()); 
  fParameters[n].SetLowerLimit(lo);
}

void MnUserTransformation::RemoveLimits(unsigned int n) {
  assert(n < fParameters.size()); 
  fParameters[n].RemoveLimits();
}

double MnUserTransformation::Value(unsigned int n) const {
  assert(n < fParameters.size()); 
  return fParameters[n].Value();
}

double MnUserTransformation::Error(unsigned int n) const {
  assert(n < fParameters.size()); 
  return fParameters[n].Error();
}

void MnUserTransformation::Fix(const char* Name) {
  Fix(Index(Name));
}

void MnUserTransformation::Release(const char* Name) {
  Release(Index(Name));
}

void MnUserTransformation::SetValue(const char* Name, double val) {
  SetValue(Index(Name), val);
}

void MnUserTransformation::SetError(const char* Name, double err) {
  SetError(Index(Name), err);
}

void MnUserTransformation::SetLimits(const char* Name, double low, double up) {
  SetLimits(Index(Name), low, up);
}

void MnUserTransformation::SetUpperLimit(const char* Name, double up) {
  SetUpperLimit(Index(Name), up);
}

void MnUserTransformation::SetLowerLimit(const char* Name, double lo) {
  SetLowerLimit(Index(Name), lo);
}

void MnUserTransformation::RemoveLimits(const char* Name) {
  RemoveLimits(Index(Name));
}

double MnUserTransformation::Value(const char* Name) const {
  return Value(Index(Name));
}

double MnUserTransformation::Error(const char* Name) const {
  return Error(Index(Name));
}
  
unsigned int MnUserTransformation::Index(const char* Name) const {
  std::vector<MinuitParameter>::const_iterator ipar = 
    std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name));
  assert(ipar != fParameters.end());
//   return (ipar - fParameters.begin());
  return (*ipar).Number();
}

const char* MnUserTransformation::Name(unsigned int n) const {
  assert(n < fParameters.size()); 
  return fParameters[n].Name();
}

  }  // namespace Minuit2

}  // namespace ROOT
