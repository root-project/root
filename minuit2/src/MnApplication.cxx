// @(#)root/minuit2:$Name:  $:$Id: MnApplication.cxx,v 1.2 2006/07/03 15:48:06 moneta Exp $
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
   // constructor from macfcn calls and tolerance 
   
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


const std::vector<MinuitParameter>& MnApplication::MinuitParameters() const {
   //access to parameters (row-wise)
   return fState.MinuitParameters();
}
//access to parameters and errors in column-wise representation 
std::vector<double> MnApplication::Params() const {return fState.Params();}
std::vector<double> MnApplication::Errors() const {return fState.Errors();}


const MinuitParameter& MnApplication::Parameter(unsigned int i) const {
   //access to single Parameter
   return fState.Parameter(i);
}


void MnApplication::Add(const char* Name, double val, double err) {
   //add free Parameter
   fState.Add(Name, val, err);
}

void MnApplication::Add(const char* Name, double val, double err, double low, double up) {
   //add limited Parameter 
   fState.Add(Name, val, err, low, up);
}

void MnApplication::Add(const char* Name, double val) {
   //add const Parameter
   fState.Add(Name, val);
}

//interaction via external number of Parameter
void MnApplication::Fix(unsigned int i) {fState.Fix(i);}
void MnApplication::Release(unsigned int i) {fState.Release(i);}
void MnApplication::SetValue(unsigned int i, double val) {
   // set value for parameter i 
   fState.SetValue(i, val);
}
void MnApplication::SetError(unsigned int i, double val) {
   // set parameter error
   fState.SetError(i, val);
}
void MnApplication::SetLimits(unsigned int i, double low, double up) {
   // set parameter limits
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
void MnApplication::SetLimits(const char* i, double low, double up) { fState.SetLimits(i, low, up);}
void MnApplication::RemoveLimits(const char* i) {fState.RemoveLimits(i);}
void MnApplication::SetPrecision(double eps) {fState.SetPrecision(eps);}

double MnApplication::Value(const char* i) const {return fState.Value(i);}
double MnApplication::Error(const char* i) const {return fState.Error(i);}


unsigned int MnApplication::Index(const char* Name) const {
   //convert Name into external number of Parameter
   return fState.Index(Name);
}

const char* MnApplication::Name(unsigned int i) const {
   //convert external number into Name of Parameter
   return fState.Name(i);
}


double MnApplication::Int2ext(unsigned int i, double val) const {
   // transformation internal -> external
   return fState.Int2ext(i, val);
}
double MnApplication::Ext2int(unsigned int e, double val) const {
   // transformation external -> internal
   return fState.Ext2int(e, val);
}
unsigned int MnApplication::IntOfExt(unsigned int ext) const {
   // get internal index for external parameter with index ext
   return fState.IntOfExt(ext);
}
unsigned int MnApplication::ExtOfInt(unsigned int internal) const { 
   // get external index for internal parameter with index internal
   return fState.ExtOfInt(internal);
}
unsigned int MnApplication::VariableParameters() const {
   // get number of variable parameters
   return fState.VariableParameters();
}


   }  // namespace Minuit2

}  // namespace ROOT
