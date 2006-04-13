// @(#)root/minuit2:$Name:  $:$Id: MnUserParameterState.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnCovarianceSqueeze.h"
#include "Minuit2/MinimumState.h"

namespace ROOT {

   namespace Minuit2 {


//
// construct from user parameters (befor minimization)
//
MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const std::vector<double>& err) : fValid(true), fCovarianceValid(false), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(MnUserParameters(par, err)), fCovariance(MnUserCovariance()), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(par), fIntCovariance(MnUserCovariance()) {}

MnUserParameterState::MnUserParameterState(const MnUserParameters& par) : fValid(true), fCovarianceValid(false), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(par), fCovariance(MnUserCovariance()), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(std::vector<double>()), fIntCovariance(MnUserCovariance()) {
  
  for(std::vector<MinuitParameter>::const_iterator ipar = MinuitParameters().begin(); ipar != MinuitParameters().end(); ipar++) {
    if((*ipar).IsConst() || (*ipar).IsFixed()) continue;
    if((*ipar).HasLimits()) 
      fIntParameters.push_back(Ext2int((*ipar).Number(), (*ipar).Value()));
    else 
      fIntParameters.push_back((*ipar).Value());
  }
}

//
// construct from user parameters + errors (befor minimization)
//
MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow) : fValid(true), fCovarianceValid(true), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(MnUserParameters()), fCovariance(MnUserCovariance(cov, nrow)), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(par), fIntCovariance(MnUserCovariance(cov, nrow)) {
  std::vector<double> err; err.reserve(par.size());
  for(unsigned int i = 0; i < par.size(); i++) {
    assert(fCovariance(i,i) > 0.);
    err.push_back(sqrt(fCovariance(i,i)));
  }
  fParameters = MnUserParameters(par, err);
  assert(fCovariance.Nrow() == VariableParameters());
}

MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const MnUserCovariance& cov) : fValid(true), fCovarianceValid(true), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(MnUserParameters()), fCovariance(cov), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(par), fIntCovariance(cov) {
  std::vector<double> err; err.reserve(par.size());
  for(unsigned int i = 0; i < par.size(); i++) {
    assert(fCovariance(i,i) > 0.);
    err.push_back(sqrt(fCovariance(i,i)));
  }
  fParameters = MnUserParameters(par, err);
  assert(fCovariance.Nrow() == VariableParameters());
}


MnUserParameterState::MnUserParameterState(const MnUserParameters& par, const MnUserCovariance& cov) : fValid(true), fCovarianceValid(true), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(par), fCovariance(cov), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(std::vector<double>()), fIntCovariance(cov) {
  fIntCovariance.Scale(0.5);
  for(std::vector<MinuitParameter>::const_iterator ipar = MinuitParameters().begin(); ipar != MinuitParameters().end(); ipar++) {
    if((*ipar).IsConst() || (*ipar).IsFixed()) continue;
    if((*ipar).HasLimits()) 
      fIntParameters.push_back(Ext2int((*ipar).Number(), (*ipar).Value()));
    else 
      fIntParameters.push_back((*ipar).Value());
  }
  assert(fCovariance.Nrow() == VariableParameters());
//
// need to Fix that in case of limited parameters
//   fIntCovariance = MnUserCovariance();
//
}

//
// construct from internal parameters (after minimization)
//
MnUserParameterState::MnUserParameterState(const MinimumState& st, double up, const MnUserTransformation& trafo) : fValid(st.IsValid()), fCovarianceValid(false), fGCCValid(false), fFVal(st.Fval()), fEDM(st.Edm()), fNFcn(st.NFcn()), fParameters(MnUserParameters()), fCovariance(MnUserCovariance()), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(std::vector<double>()), fIntCovariance(MnUserCovariance()) {

  for(std::vector<MinuitParameter>::const_iterator ipar = trafo.Parameters().begin(); ipar != trafo.Parameters().end(); ipar++) {
    if((*ipar).IsConst()) {
      Add((*ipar).Name(), (*ipar).Value());
    } else if((*ipar).IsFixed()) {
      Add((*ipar).Name(), (*ipar).Value(), (*ipar).Error());
      if((*ipar).HasLimits()) {
	if((*ipar).HasLowerLimit() && (*ipar).HasUpperLimit())
	  SetLimits((*ipar).Name(), (*ipar).LowerLimit(),(*ipar).UpperLimit());
	else if((*ipar).HasLowerLimit() && !(*ipar).HasUpperLimit())
	  SetLowerLimit((*ipar).Name(), (*ipar).LowerLimit());
	else
	  SetUpperLimit((*ipar).Name(), (*ipar).UpperLimit());
      }
      Fix((*ipar).Name());
    } else if((*ipar).HasLimits()) {
      unsigned int i = trafo.IntOfExt((*ipar).Number());
      double err = st.HasCovariance() ? sqrt(2.*up*st.Error().InvHessian()(i,i)) : st.Parameters().Dirin()(i);
      Add((*ipar).Name(), trafo.Int2ext(i, st.Vec()(i)), trafo.Int2extError(i, st.Vec()(i), err));
      if((*ipar).HasLowerLimit() && (*ipar).HasUpperLimit())
	SetLimits((*ipar).Name(), (*ipar).LowerLimit(), (*ipar).UpperLimit());
      else if((*ipar).HasLowerLimit() && !(*ipar).HasUpperLimit())
	SetLowerLimit((*ipar).Name(), (*ipar).LowerLimit());
      else
	SetUpperLimit((*ipar).Name(), (*ipar).UpperLimit());
    } else {
      unsigned int i = trafo.IntOfExt((*ipar).Number());
      double err = st.HasCovariance() ? sqrt(2.*up*st.Error().InvHessian()(i,i)) : st.Parameters().Dirin()(i);
      Add((*ipar).Name(), st.Vec()(i), err);
    }
  }

  fCovarianceValid = st.Error().IsValid();

  if(fCovarianceValid) {
    fCovariance = trafo.Int2extCovariance(st.Vec(), st.Error().InvHessian());
    fIntCovariance = MnUserCovariance(std::vector<double>(st.Error().InvHessian().Data(), st.Error().InvHessian().Data()+st.Error().InvHessian().size()), st.Error().InvHessian().Nrow());
    fCovariance.Scale(2.*up);
    fGlobalCC = MnGlobalCorrelationCoeff(st.Error().InvHessian());
    fGCCValid = fGlobalCC.IsValid();

    assert(fCovariance.Nrow() == VariableParameters());
  }
}

// facade: forward interface of MnUserParameters and MnUserTransformation
// via MnUserParameterState

//access to parameters (row-wise)
const std::vector<MinuitParameter>& MnUserParameterState::MinuitParameters() const {
  return fParameters.Parameters();
}
//access to parameters and errors in column-wise representation 
std::vector<double> MnUserParameterState::Params() const {
  return fParameters.Params();
}
std::vector<double> MnUserParameterState::Errors() const {
  return fParameters.Errors();
}

//access to single Parameter
const MinuitParameter& MnUserParameterState::Parameter(unsigned int i) const {
  return fParameters.Parameter(i);
}

//add free Parameter
void MnUserParameterState::Add(const char* Name, double val, double err) {
  if ( fParameters.Add(Name, val, err) ) { 
    fIntParameters.push_back(val);
    fCovarianceValid = false;
    fGCCValid = false;
    fValid = true;
  }
  else { 
    int i = Index(Name);
    SetValue(i,val);
    SetError(i,err);
  }
    
}

//add limited Parameter
void MnUserParameterState::Add(const char* Name, double val, double err, double low, double up) {
  if ( fParameters.Add(Name, val, err, low, up) ) {  
    fCovarianceValid = false;
    fIntParameters.push_back(Ext2int(Index(Name), val));
    fGCCValid = false;
    fValid = true;
  }
  else { // Parameter already exist - just set values
    int i = Index(Name);
    SetValue(i,val);
    SetError(i,err);
    SetLimits(i,low,up);
  }
    
    
}

//add const Parameter
void MnUserParameterState::Add(const char* Name, double val) {
  if ( fParameters.Add(Name, val) ) 
    fValid = true;
  else 
    SetValue(Name,val);
}

//interaction via external number of Parameter
void MnUserParameterState::Fix(unsigned int e) {
  unsigned int i = IntOfExt(e);
  if(fCovarianceValid) {
    fCovariance = MnCovarianceSqueeze()(fCovariance, i);
    fIntCovariance = MnCovarianceSqueeze()(fIntCovariance, i);
  }
  fIntParameters.erase(fIntParameters.begin()+i, fIntParameters.begin()+i+1);  
  fParameters.Fix(e);
  fGCCValid = false;
}

void MnUserParameterState::Release(unsigned int e) {
  fParameters.Release(e);
  fCovarianceValid = false;
  fGCCValid = false;
  unsigned int i = IntOfExt(e);
  if(Parameter(e).HasLimits())
    fIntParameters.insert(fIntParameters.begin()+i, Ext2int(e, Parameter(e).Value()));    
  else
    fIntParameters.insert(fIntParameters.begin()+i, Parameter(e).Value());
}

void MnUserParameterState::SetValue(unsigned int e, double val) {
  fParameters.SetValue(e, val);
  if(!Parameter(e).IsFixed() && !Parameter(e).IsConst()) {
    unsigned int i = IntOfExt(e);
    if(Parameter(e).HasLimits())
      fIntParameters[i] = Ext2int(e, val);
    else
      fIntParameters[i] = val;
  }
}

void MnUserParameterState::SetError(unsigned int e, double val) {
  fParameters.SetError(e, val);
}

void MnUserParameterState::SetLimits(unsigned int e, double low, double up) {
  fParameters.SetLimits(e, low, up);
  fCovarianceValid = false;
  fGCCValid = false;
  if(!Parameter(e).IsFixed() && !Parameter(e).IsConst()) {
    unsigned int i = IntOfExt(e);
    if(low < fIntParameters[i] && fIntParameters[i] < up)
      fIntParameters[i] = Ext2int(e, fIntParameters[i]);
    else
      fIntParameters[i] = Ext2int(e, 0.5*(low+up));
  }
}

void MnUserParameterState::SetUpperLimit(unsigned int e, double up) {
  fParameters.SetUpperLimit(e, up);
  fCovarianceValid = false;
  fGCCValid = false;
  if(!Parameter(e).IsFixed() && !Parameter(e).IsConst()) {
    unsigned int i = IntOfExt(e);
    if(fIntParameters[i] < up)
      fIntParameters[i] = Ext2int(e, fIntParameters[i]);
    else
      fIntParameters[i] = Ext2int(e, up - 0.5*fabs(up + 1.));
  }
}

void MnUserParameterState::SetLowerLimit(unsigned int e, double low) {
  fParameters.SetLowerLimit(e, low);
  fCovarianceValid = false;
  fGCCValid = false;
  if(!Parameter(e).IsFixed() && !Parameter(e).IsConst()) {
    unsigned int i = IntOfExt(e);
    if(low < fIntParameters[i])
      fIntParameters[i] = Ext2int(e, fIntParameters[i]);
    else
      fIntParameters[i] = Ext2int(e, low + 0.5*fabs(low + 1.));
  }
}

void MnUserParameterState::RemoveLimits(unsigned int e) {
  fParameters.RemoveLimits(e);
  fCovarianceValid = false;
  fGCCValid = false;
  if(!Parameter(e).IsFixed() && !Parameter(e).IsConst())
    fIntParameters[IntOfExt(e)] = Value(e);  
}

double MnUserParameterState::Value(unsigned int i) const {
  return fParameters.Value(i);
}
double MnUserParameterState::Error(unsigned int i) const {
  return fParameters.Error(i);
}
  
//interaction via Name of Parameter
void MnUserParameterState::Fix(const char* Name) {
  Fix(Index(Name));
}

void MnUserParameterState::Release(const char* Name) {
  Release(Index(Name));
}

void MnUserParameterState::SetValue(const char* Name, double val) {
  SetValue(Index(Name), val);
}

void MnUserParameterState::SetError(const char* Name, double val) {
  SetError(Index(Name), val);
}

void MnUserParameterState::SetLimits(const char* Name, double low, double up) {
  SetLimits(Index(Name), low, up);
}

void MnUserParameterState::SetUpperLimit(const char* Name, double up) {
  SetUpperLimit(Index(Name), up);
}

void MnUserParameterState::SetLowerLimit(const char* Name, double low) {
  SetLowerLimit(Index(Name), low);
}

void MnUserParameterState::RemoveLimits(const char* Name) {
  RemoveLimits(Index(Name));
}

double MnUserParameterState::Value(const char* Name) const {
  return Value(Index(Name));
}
double MnUserParameterState::Error(const char* Name) const {
  return Error(Index(Name));
}
  
//convert Name into external number of Parameter
unsigned int MnUserParameterState::Index(const char* Name) const {
  return fParameters.Index(Name);
}
//convert external number into Name of Parameter
const char* MnUserParameterState::Name(unsigned int i) const {
  return fParameters.Name(i);
}

// transformation internal <-> external
double MnUserParameterState::Int2ext(unsigned int i, double val) const {
  return fParameters.Trafo().Int2ext(i, val);
}
double MnUserParameterState::Ext2int(unsigned int e, double val) const {
  return fParameters.Trafo().Ext2int(e, val);
}
unsigned int MnUserParameterState::IntOfExt(unsigned int ext) const {
  return fParameters.Trafo().IntOfExt(ext);
}
unsigned int MnUserParameterState::ExtOfInt(unsigned int internal) const { 
    return fParameters.Trafo().ExtOfInt(internal);
}
unsigned int MnUserParameterState::VariableParameters() const {
  return fParameters.Trafo().VariableParameters();
}
const MnMachinePrecision& MnUserParameterState::Precision() const {
   return fParameters.Precision();
}

void MnUserParameterState::SetPrecision(double eps) {
  fParameters.SetPrecision(eps);
}

  }  // namespace Minuit2

}  // namespace ROOT
