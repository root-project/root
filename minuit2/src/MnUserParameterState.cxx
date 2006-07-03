// @(#)root/minuit2:$Name:  $:$Id: MnUserParameterState.cxx,v 1.2 2006/04/13 08:39:23 moneta Exp $
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
MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const std::vector<double>& err) : 
   fValid(true), fCovarianceValid(false), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(MnUserParameters(par, err)), fCovariance(MnUserCovariance()), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(par), fIntCovariance(MnUserCovariance()) 
      {}

MnUserParameterState::MnUserParameterState(const MnUserParameters& par) : 
   fValid(true), fCovarianceValid(false), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(par), fCovariance(MnUserCovariance()), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(std::vector<double>()), fIntCovariance(MnUserCovariance()) {
   // construct from user parameters (befor minimization)
  
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
   // construct from user parameters + errors (before minimization) using std::vectors

   std::vector<double> err; err.reserve(par.size());
   for(unsigned int i = 0; i < par.size(); i++) {
      assert(fCovariance(i,i) > 0.);
      err.push_back(sqrt(fCovariance(i,i)));
   }
   fParameters = MnUserParameters(par, err);
   assert(fCovariance.Nrow() == VariableParameters());
}

MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const MnUserCovariance& cov) : 
   fValid(true), fCovarianceValid(true), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(MnUserParameters()), fCovariance(cov), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(par), fIntCovariance(cov) {
   //construct from user parameters + errors (befor minimization) using std::vector (params) and MnUserCovariance class

   std::vector<double> err; err.reserve(par.size());
   for(unsigned int i = 0; i < par.size(); i++) {
      assert(fCovariance(i,i) > 0.);
      err.push_back(sqrt(fCovariance(i,i)));
   }
   fParameters = MnUserParameters(par, err);
   assert(fCovariance.Nrow() == VariableParameters());
}


MnUserParameterState::MnUserParameterState(const MnUserParameters& par, const MnUserCovariance& cov) : 
   fValid(true), fCovarianceValid(true), fGCCValid(false), fFVal(0.), fEDM(0.), fNFcn(0), fParameters(par), fCovariance(cov), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(std::vector<double>()), fIntCovariance(cov) {
   //construct from user parameters + errors (befor minimization) using MnUserParameters and MnUserCovariance classes

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
//
MnUserParameterState::MnUserParameterState(const MinimumState& st, double up, const MnUserTransformation& trafo) : 
   fValid(st.IsValid()), fCovarianceValid(false), fGCCValid(false), fFVal(st.Fval()), fEDM(st.Edm()), fNFcn(st.NFcn()), fParameters(MnUserParameters()), fCovariance(MnUserCovariance()), fGlobalCC(MnGlobalCorrelationCoeff()), fIntParameters(std::vector<double>()), fIntCovariance(MnUserCovariance()) {
   //
   // construct from internal parameters (after minimization)
   //
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


const std::vector<MinuitParameter>& MnUserParameterState::MinuitParameters() const {
   //access to parameters (row-wise)
   return fParameters.Parameters();
}
 
std::vector<double> MnUserParameterState::Params() const {
   //access to parameters in column-wise representation
   return fParameters.Params();
}
std::vector<double> MnUserParameterState::Errors() const {
   //access to errors in column-wise representation
   return fParameters.Errors();
}

const MinuitParameter& MnUserParameterState::Parameter(unsigned int i) const {
   //access to single Parameter i
   return fParameters.Parameter(i);
}

void MnUserParameterState::Add(const char* Name, double val, double err) {
   //add free Parameter
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

void MnUserParameterState::Add(const char* Name, double val, double err, double low, double up) {
   //add limited Parameter
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

void MnUserParameterState::Add(const char* Name, double val) {
   //add const Parameter
   if ( fParameters.Add(Name, val) ) 
      fValid = true;
   else 
      SetValue(Name,val);
}

//interaction via external number of Parameter

void MnUserParameterState::Fix(unsigned int e) {
   // fix parameter e (external index)
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
   // release parameter e (external index)
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
   // set error for  parameter e (external index)
   fParameters.SetError(e, val);
}

void MnUserParameterState::SetLimits(unsigned int e, double low, double up) {
   // set limits for parameter e (external index)
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
   // set upper limit for parameter e (external index)
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
   // set lower limit for parameter e (external index)
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
   // remove limit for parameter e (external index)
   fParameters.RemoveLimits(e);
   fCovarianceValid = false;
   fGCCValid = false;
   if(!Parameter(e).IsFixed() && !Parameter(e).IsConst())
      fIntParameters[IntOfExt(e)] = Value(e);  
}

double MnUserParameterState::Value(unsigned int i) const {
   // get value for parameter e (external index)
   return fParameters.Value(i);
}
double MnUserParameterState::Error(unsigned int i) const {
   // get error for parameter e (external index)
   return fParameters.Error(i);
}

//interaction via Name of Parameter

void MnUserParameterState::Fix(const char* Name) { Fix(Index(Name));}

void MnUserParameterState::Release(const char* Name) {Release(Index(Name));}

void MnUserParameterState::SetValue(const char* Name, double val) {SetValue(Index(Name), val);}

void MnUserParameterState::SetError(const char* Name, double val) { SetError(Index(Name), val);}

void MnUserParameterState::SetLimits(const char* Name, double low, double up) {SetLimits(Index(Name), low, up);}

void MnUserParameterState::SetUpperLimit(const char* Name, double up) { SetUpperLimit(Index(Name), up);}

void MnUserParameterState::SetLowerLimit(const char* Name, double low) {SetLowerLimit(Index(Name), low);}

void MnUserParameterState::RemoveLimits(const char* Name) {RemoveLimits(Index(Name));}

double MnUserParameterState::Value(const char* Name) const {return Value(Index(Name));}

double MnUserParameterState::Error(const char* Name) const {return Error(Index(Name));}


unsigned int MnUserParameterState::Index(const char* Name) const {
   //convert Name into external number of Parameter
   return fParameters.Index(Name);
}

const char* MnUserParameterState::Name(unsigned int i) const {
   //convert external number into Name of Parameter
   return fParameters.Name(i);
}

// transformation internal <-> external (forward to transformation class)

double MnUserParameterState::Int2ext(unsigned int i, double val) const {
   // internal to external value
   return fParameters.Trafo().Int2ext(i, val);
}
double MnUserParameterState::Ext2int(unsigned int e, double val) const {
    // external  to internal value 
   return fParameters.Trafo().Ext2int(e, val);
}
unsigned int MnUserParameterState::IntOfExt(unsigned int ext) const {
   // return internal index for external index ext 
   return fParameters.Trafo().IntOfExt(ext);
}
unsigned int MnUserParameterState::ExtOfInt(unsigned int internal) const { 
    // return external index for internal index internal
   return fParameters.Trafo().ExtOfInt(internal);
}
unsigned int MnUserParameterState::VariableParameters() const {
   // return number of variable parameters
   return fParameters.Trafo().VariableParameters();
}
const MnMachinePrecision& MnUserParameterState::Precision() const {
   // return global parameter precision
   return fParameters.Precision();
}

void MnUserParameterState::SetPrecision(double eps) {
   // set global parameter precision
   fParameters.SetPrecision(eps);
}

   }  // namespace Minuit2

}  // namespace ROOT
