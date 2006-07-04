// @(#)root/minuit2:$Name:  $:$Id: MnUserParameters.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
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

// all implemented forwarding to MnUserTransformation class

const std::vector<MinuitParameter>& MnUserParameters::Parameters() const {
   // return vector of MinuitParameter objects
   return fTransformation.Parameters();
}

std::vector<double> MnUserParameters::Params() const {
   // return std::vector of double with parameter values 
   return fTransformation.Params();
}

std::vector<double> MnUserParameters::Errors() const {
   // return std::vector of double with parameter errors
   return fTransformation.Errors();
}

const MinuitParameter& MnUserParameters::Parameter(unsigned int n) const {
   // return the MinuitParameter object for index n (external)
   return fTransformation.Parameter(n);
}

bool MnUserParameters::Add(const char* Name, double val, double err) {
   // add a new unlimited parameter giving name, value and err (step size)
   // return false if parameter already exists
   return fTransformation.Add(Name, val, err);
}

bool  MnUserParameters::Add(const char* Name, double val, double err, double low, double up) {
   // add a new limited parameter giving name, value, err (step size) and lower/upper limits
   // return false if parameter already exists
   return fTransformation.Add(Name, val, err, low, up);
}

bool  MnUserParameters::Add(const char* Name, double val) {
   // add a new unlimited parameter giving name and value
   // return false if parameter already exists
   return fTransformation.Add(Name, val);
}

void MnUserParameters::Fix(unsigned int n) {
   // fix parameter n
   fTransformation.Fix(n);
}

void MnUserParameters::Release(unsigned int n) {
   // release parameter n
   fTransformation.Release(n);
}

void MnUserParameters::SetValue(unsigned int n, double val) {
   // set value for parameter n
   fTransformation.SetValue(n, val);
}

void MnUserParameters::SetError(unsigned int n, double err) {
   // set error for parameter n
   fTransformation.SetError(n, err);
}

void MnUserParameters::SetLimits(unsigned int n, double low, double up) {
   // set limits (lower/upper) for parameter n
   fTransformation.SetLimits(n, low, up);
}

void MnUserParameters::SetUpperLimit(unsigned int n, double up) {
   // set upper limit for parameter n
   fTransformation.SetUpperLimit(n, up);
}

void MnUserParameters::SetLowerLimit(unsigned int n, double low) {
   // set lower limit for parameter n
   fTransformation.SetLowerLimit(n, low);
}

void MnUserParameters::RemoveLimits(unsigned int n) {
   // remove limits for parameter n
   fTransformation.RemoveLimits(n);
}

double MnUserParameters::Value(unsigned int n) const {
   // get value for parameter n
   return fTransformation.Value(n);
}

double MnUserParameters::Error(unsigned int n) const {
   // get error for parameter n
   return fTransformation.Error(n);
}

// interface using  parameter name

void MnUserParameters::Fix(const char* Name) {
   // fix parameter 
   Fix(Index(Name));
}

void MnUserParameters::Release(const char* Name) {
   // release parameter 
   Release(Index(Name));
}

void MnUserParameters::SetValue(const char* Name, double val) {
   // set value for parameter 
   SetValue(Index(Name), val);
}

void MnUserParameters::SetError(const char* Name, double err) {
   // set error
   SetError(Index(Name), err);
}

void MnUserParameters::SetLimits(const char* Name, double low, double up) {
   // set lower/upper limits
   SetLimits(Index(Name), low, up);
}

void MnUserParameters::SetUpperLimit(const char* Name, double up) {
   // set upper limit
   fTransformation.SetUpperLimit(Index(Name), up);
}

void MnUserParameters::SetLowerLimit(const char* Name, double low) {
   // set lower limit
   fTransformation.SetLowerLimit(Index(Name), low);
}

void MnUserParameters::RemoveLimits(const char* Name) {
   // remove limits
   RemoveLimits(Index(Name));
}

double MnUserParameters::Value(const char* Name) const {
   // get parameter value
   return Value(Index(Name));
}

double MnUserParameters::Error(const char* Name) const {
   // get parameter error
   return Error(Index(Name));
}

unsigned int MnUserParameters::Index(const char* Name) const {
   // get index (external) corresponding to Name
   return fTransformation.Index(Name);
}

const char* MnUserParameters::Name(unsigned int n) const {
   // get Name corresponding to index (external)
   return fTransformation.Name(n);
}

const MnMachinePrecision& MnUserParameters::Precision() const {
   // get global paramter precision
   return fTransformation.Precision();
}

   }  // namespace Minuit2

}  // namespace ROOT
