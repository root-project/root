// @(#)root/minuit2:$Name:  $:$Id: MnUserParameters.cpp,v 1.8.2.4 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnUserParameters.h"

namespace ROOT {

   namespace Minuit2 {


MnUserParameters::MnUserParameters(const std::vector<double>& par, const std::vector<double>& err) : fTransformation(par, err) {}

const std::vector<MinuitParameter>& MnUserParameters::Parameters() const {
  return fTransformation.Parameters();
}

std::vector<double> MnUserParameters::Params() const {
  return fTransformation.Params();
}

std::vector<double> MnUserParameters::Errors() const {
  return fTransformation.Errors();
}

const MinuitParameter& MnUserParameters::Parameter(unsigned int n) const {
  return fTransformation.Parameter(n);
}

bool MnUserParameters::Add(const char* Name, double val, double err) {
  return fTransformation.Add(Name, val, err);
}

bool  MnUserParameters::Add(const char* Name, double val, double err, double low, double up) {
  return fTransformation.Add(Name, val, err, low, up);
}

bool  MnUserParameters::Add(const char* Name, double val) {
  return fTransformation.Add(Name, val);
}

void MnUserParameters::Fix(unsigned int n) {
  fTransformation.Fix(n);
}

void MnUserParameters::Release(unsigned int n) {
  fTransformation.Release(n);
}

void MnUserParameters::SetValue(unsigned int n, double val) {
  fTransformation.SetValue(n, val);
}

void MnUserParameters::SetError(unsigned int n, double err) {
  fTransformation.SetError(n, err);
}

void MnUserParameters::SetLimits(unsigned int n, double low, double up) {
  fTransformation.SetLimits(n, low, up);
}

void MnUserParameters::SetUpperLimit(unsigned int n, double up) {
  fTransformation.SetUpperLimit(n, up);
}

void MnUserParameters::SetLowerLimit(unsigned int n, double low) {
  fTransformation.SetLowerLimit(n, low);
}

void MnUserParameters::RemoveLimits(unsigned int n) {
  fTransformation.RemoveLimits(n);
}

double MnUserParameters::Value(unsigned int n) const {
  return fTransformation.Value(n);
}

double MnUserParameters::Error(unsigned int n) const {
  return fTransformation.Error(n);
}

void MnUserParameters::Fix(const char* Name) {
  Fix(Index(Name));
}

void MnUserParameters::Release(const char* Name) {
  Release(Index(Name));
}

void MnUserParameters::SetValue(const char* Name, double val) {
  SetValue(Index(Name), val);
}

void MnUserParameters::SetError(const char* Name, double err) {
  SetError(Index(Name), err);
}

void MnUserParameters::SetLimits(const char* Name, double low, double up) {
  SetLimits(Index(Name), low, up);
}

void MnUserParameters::SetUpperLimit(const char* Name, double up) {
  fTransformation.SetUpperLimit(Index(Name), up);
}

void MnUserParameters::SetLowerLimit(const char* Name, double low) {
  fTransformation.SetLowerLimit(Index(Name), low);
}

void MnUserParameters::RemoveLimits(const char* Name) {
  RemoveLimits(Index(Name));
}

double MnUserParameters::Value(const char* Name) const {
  return Value(Index(Name));
}

double MnUserParameters::Error(const char* Name) const {
  return Error(Index(Name));
}
  
unsigned int MnUserParameters::Index(const char* Name) const {
  return fTransformation.Index(Name);
}

const char* MnUserParameters::Name(unsigned int n) const {
  return fTransformation.Name(n);
}

const MnMachinePrecision& MnUserParameters::Precision() const {
  return fTransformation.Precision();
}

  }  // namespace Minuit2

}  // namespace ROOT
