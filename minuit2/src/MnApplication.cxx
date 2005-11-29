// @(#)root/minuit2:$Name:  $:$Id: MnApplication.cpp,v 1.2.4.4 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnApplication.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/ModularFunctionMinimizer.h"

namespace ROOT {

   namespace Minuit2 {


FunctionMinimum MnApplication::operator()(unsigned int maxfcn, double toler) {

  assert(fState.IsValid());
  unsigned int npar = VariableParameters();
//   assert(npar > 0);
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;
  FunctionMinimum min = Minimizer().Minimize( Fcnbase(), fState, fStrategy, maxfcn, toler);
  fNumCall += min.NFcn();
  fState = min.UserState();
  return min;
}

// facade: forward interface of MnUserParameters and MnUserTransformation
// via MnUserParameterState

//access to parameters (row-wise)
const std::vector<MinuitParameter>& MnApplication::MinuitParameters() const {
  return fState.MinuitParameters();
}
//access to parameters and errors in column-wise representation 
std::vector<double> MnApplication::Params() const {return fState.Params();}
std::vector<double> MnApplication::Errors() const {return fState.Errors();}

//access to single Parameter
const MinuitParameter& MnApplication::Parameter(unsigned int i) const {
  return fState.Parameter(i);
}

//add free Parameter
void MnApplication::Add(const char* Name, double val, double err) {
  fState.Add(Name, val, err);
}
//add limited Parameter
void MnApplication::Add(const char* Name, double val, double err, double low, double up) {
  fState.Add(Name, val, err, low, up);
}
//add const Parameter
void MnApplication::Add(const char* Name, double val) {
  fState.Add(Name, val);
}

//interaction via external number of Parameter
void MnApplication::Fix(unsigned int i) {fState.Fix(i);}
void MnApplication::Release(unsigned int i) {fState.Release(i);}
void MnApplication::SetValue(unsigned int i, double val) {
  fState.SetValue(i, val);
}
void MnApplication::SetError(unsigned int i, double val) {
  fState.SetError(i, val);
}
void MnApplication::SetLimits(unsigned int i, double low, double up) {
  fState.SetLimits(i, low, up);
}
void MnApplication::RemoveLimits(unsigned int i) {fState.RemoveLimits(i);}

double MnApplication::Value(unsigned int i) const {return fState.Value(i);}
double MnApplication::Error(unsigned int i) const {return fState.Error(i);}
  
//interaction via Name of Parameter
void MnApplication::Fix(const char* i) {fState.Fix(i);}
void MnApplication::Release(const char* i) {fState.Release(i);}
void MnApplication::SetValue(const char* i, double val) {fState.SetValue(i, val);}
void MnApplication::SetError(const char* i, double val) {fState.SetError(i, val);}
void MnApplication::SetLimits(const char* i, double low, double up) {
  fState.SetLimits(i, low, up);
}
void MnApplication::RemoveLimits(const char* i) {fState.RemoveLimits(i);}
void MnApplication::SetPrecision(double eps) {fState.SetPrecision(eps);}

double MnApplication::Value(const char* i) const {return fState.Value(i);}
double MnApplication::Error(const char* i) const {return fState.Error(i);}
  
//convert Name into external number of Parameter
unsigned int MnApplication::Index(const char* Name) const {
  return fState.Index(Name);
}
//convert external number into Name of Parameter
const char* MnApplication::Name(unsigned int i) const {
  return fState.Name(i);
}

// transformation internal <-> external
double MnApplication::Int2ext(unsigned int i, double val) const {
  return fState.Int2ext(i, val);
}
double MnApplication::Ext2int(unsigned int e, double val) const {
  return fState.Ext2int(e, val);
}
unsigned int MnApplication::IntOfExt(unsigned int ext) const {
  return fState.IntOfExt(ext);
}
unsigned int MnApplication::ExtOfInt(unsigned int internal) const { 
    return fState.ExtOfInt(internal);
}
unsigned int MnApplication::VariableParameters() const {
  return fState.VariableParameters();
}


  }  // namespace Minuit2

}  // namespace ROOT
