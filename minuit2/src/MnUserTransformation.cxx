// @(#)root/minuit2:$Name:  $:$Id: MnUserTransformation.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
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
   // constructor from a vector of parameter values and a vector of errors (step  sizes) 
   // class has as datga member the transformation objects (all of the types), 
   // the std::vector of MinuitParameter objects and the vector with the index conversions from 
   // internal to external (fExtOfInt)
   
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
   // transform an internal  Minuit vector of internal values in a std::vector of external values 
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
   // return external value from internal value for parameter i
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
   // return external error from internal error for parameter i

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
   // return the external covariance matrix from the internal error matrix and the internal parameter value
   // the vector of internal parameter is needed for the derivaties (Jacobian of the transformation)
   // Vext(i,j) = Vint(i,j) * dPext(i)/dPint(i) * dPext(j)/dPint(j) 
   
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
   // return the external value for parameter i with value val
   
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
   // return the derivative of the int->ext transformation: dPext(i) / dPint(i)
   // for the parameter i with value val

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
   // return internal index given external one ext
   assert(ext < fParameters.size());
   assert(!fParameters[ext].IsFixed());
   assert(!fParameters[ext].IsConst());
   std::vector<unsigned int>::const_iterator iind = std::find(fExtOfInt.begin(), fExtOfInt.end(), ext);
   assert(iind != fExtOfInt.end());
   
   return (iind - fExtOfInt.begin());  
}

std::vector<double> MnUserTransformation::Params() const {
   // return std::vector of double with parameter values 
   std::vector<double> result; result.reserve(fParameters.size());
   for(std::vector<MinuitParameter>::const_iterator ipar = Parameters().begin();
       ipar != Parameters().end(); ipar++)
      result.push_back((*ipar).Value());
   
   return result;
}

std::vector<double> MnUserTransformation::Errors() const {
   // return std::vector of double with parameter errors
   std::vector<double> result; result.reserve(fParameters.size());
   for(std::vector<MinuitParameter>::const_iterator ipar = Parameters().begin();
       ipar != Parameters().end(); ipar++)
      result.push_back((*ipar).Error());
   
   return result;
}

const MinuitParameter& MnUserTransformation::Parameter(unsigned int n) const {
   // return the MinuitParameter object for index n (external)
   assert(n < fParameters.size()); 
   return fParameters[n];
}

bool MnUserTransformation::Add(const char* Name, double val, double err) {
   // add a new unlimited parameter giving name, value and err (step size)
   // return false if parameter already exists
   // return false if parameter already exists
   if (std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name)) != fParameters.end() ) 
      return false; 
   fExtOfInt.push_back(fParameters.size());
   fCache.push_back(val);
   fParameters.push_back(MinuitParameter(fParameters.size(), Name, val, err));
   return true;
}

bool MnUserTransformation::Add(const char* Name, double val, double err, double low, double up) {
   // add a new limited parameter giving name, value, err (step size) and lower/upper limits
   // return false if parameter already exists
   if (std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name)) != fParameters.end() ) 
      return false; 
   fExtOfInt.push_back(fParameters.size());
   fCache.push_back(val);
   fParameters.push_back(MinuitParameter(fParameters.size(), Name, val, err, low, up));
   return true;
}

bool MnUserTransformation::Add(const char* Name, double val) {
   // add a new unlimited parameter giving name and value
   // return false if parameter already exists
   if (std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name)) != fParameters.end() ) 
      return false; 
   fCache.push_back(val);
   fParameters.push_back(MinuitParameter(fParameters.size(), Name, val));
   return true;
}

void MnUserTransformation::Fix(unsigned int n) {
  // fix parameter n (external index)
   assert(n < fParameters.size()); 
   std::vector<unsigned int>::iterator iind = std::find(fExtOfInt.begin(), fExtOfInt.end(), n);
   assert(iind != fExtOfInt.end());
   fExtOfInt.erase(iind, iind+1);
   fParameters[n].Fix();
}

void MnUserTransformation::Release(unsigned int n) {
   // release parameter n (external index)
   assert(n < fParameters.size()); 
   std::vector<unsigned int>::const_iterator iind = std::find(fExtOfInt.begin(), fExtOfInt.end(), n);
   assert(iind == fExtOfInt.end());
   fExtOfInt.push_back(n);
   std::sort(fExtOfInt.begin(), fExtOfInt.end());
   fParameters[n].Release();
}

void MnUserTransformation::SetValue(unsigned int n, double val) {
   // set value for parameter n (external index)
   assert(n < fParameters.size()); 
   fParameters[n].SetValue(val);
   fCache[n] = val;
}

void MnUserTransformation::SetError(unsigned int n, double err) {
   // set error for parameter n (external index)
   assert(n < fParameters.size()); 
   fParameters[n].SetError(err);
}

void MnUserTransformation::SetLimits(unsigned int n, double low, double up) {
   // set limits (lower/upper) for parameter n (external index)
   assert(n < fParameters.size());
   assert(low != up);
   fParameters[n].SetLimits(low, up);
}

void MnUserTransformation::SetUpperLimit(unsigned int n, double up) {
   // set upper limit for parameter n (external index)
   assert(n < fParameters.size()); 
   fParameters[n].SetUpperLimit(up);
}

void MnUserTransformation::SetLowerLimit(unsigned int n, double lo) {
   // set lower limit for parameter n (external index)
   assert(n < fParameters.size()); 
   fParameters[n].SetLowerLimit(lo);
}

void MnUserTransformation::RemoveLimits(unsigned int n) {
   // remove limits for parameter n (external index)
   assert(n < fParameters.size()); 
   fParameters[n].RemoveLimits();
}

double MnUserTransformation::Value(unsigned int n) const {
   // get value for parameter n (external index)
   assert(n < fParameters.size()); 
   return fParameters[n].Value();
}

double MnUserTransformation::Error(unsigned int n) const {
   // get error for parameter n (external index)
   assert(n < fParameters.size()); 
   return fParameters[n].Error();
}

// interface by parameter name

void MnUserTransformation::Fix(const char* Name) {
   // fix parameter 
   Fix(Index(Name));
}

void MnUserTransformation::Release(const char* Name) {
   // release parameter 
   Release(Index(Name));
}

void MnUserTransformation::SetValue(const char* Name, double val) {
   // set value for parameter 
   SetValue(Index(Name), val);
}

void MnUserTransformation::SetError(const char* Name, double err) {
   // set error
   SetError(Index(Name), err);
}

void MnUserTransformation::SetLimits(const char* Name, double low, double up) {
   // set lower/upper limits
   SetLimits(Index(Name), low, up);
}

void MnUserTransformation::SetUpperLimit(const char* Name, double up) {
   // set upper limit
   SetUpperLimit(Index(Name), up);
}

void MnUserTransformation::SetLowerLimit(const char* Name, double lo) {
   // set lower limit
   SetLowerLimit(Index(Name), lo);
}

void MnUserTransformation::RemoveLimits(const char* Name) {
   // remove limits
   RemoveLimits(Index(Name));
}

double MnUserTransformation::Value(const char* Name) const {
   // get parameter value
   return Value(Index(Name));
}

double MnUserTransformation::Error(const char* Name) const {
   // get parameter error
   return Error(Index(Name));
}

unsigned int MnUserTransformation::Index(const char* Name) const {
   // get index (external) corresponding to Name
   std::vector<MinuitParameter>::const_iterator ipar = 
   std::find_if(fParameters.begin(), fParameters.end(), MnParStr(Name));
   assert(ipar != fParameters.end());
   //   return (ipar - fParameters.begin());
   return (*ipar).Number();
}

const char* MnUserTransformation::Name(unsigned int n) const {
   // get Name corresponding to index (external)
   assert(n < fParameters.size()); 
   return fParameters[n].Name();
}

   }  // namespace Minuit2

}  // namespace ROOT
