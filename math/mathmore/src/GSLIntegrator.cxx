// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Implementation file for class GSLIntegrator
//
// Created by: Lorenzo Moneta  at Thu Nov 11 14:22:32 2004
//
// Last update: Thu Nov 11 14:22:32 2004
//

#include "Math/GSLIntegrator.h"

#include "gsl/gsl_integration.h"

#include "Math/IFunction.h"
#include "GSLIntegrationWorkspace.h"
#include "GSLFunctionWrapper.h"

// for toupper
#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here


#include <iostream>



namespace ROOT {
namespace Math {




GSLIntegrator::GSLIntegrator(const Integration::Type type , const Integration::GKRule rule, double absTol, double relTol, size_t size) :
   fType(type),
   fRule(rule),
   fAbsTol(absTol),
   fRelTol(relTol),
   fSize(size),
   fMaxIntervals(size),
   fResult(0),fError(0),fStatus(-1),fNEval(-1),
   fFunction(0),
   fWorkspace(0)
{
   // constructor for all types of integrations
   // allocate workspace (only if not adaptive algorithm)
   if (type !=  Integration::kNONADAPTIVE)
      fWorkspace = new GSLIntegrationWorkspace( fSize);


}



GSLIntegrator::GSLIntegrator(double absTol, double relTol, size_t size) :
   fType(Integration::kADAPTIVESINGULAR),
   fRule(Integration::kGAUSS31),
   fAbsTol(absTol),
   fRelTol(relTol),
   fSize(size),
   fMaxIntervals(size),
   fResult(0),fError(0),fStatus(-1),fNEval(-1),
   fFunction(0),
   fWorkspace(0)
{
   // constructor with default type (ADaptiveSingular) ,  rule is not needed
   fWorkspace = new GSLIntegrationWorkspace( fSize);

}



GSLIntegrator::GSLIntegrator(const Integration::Type type , double absTol, double relTol, size_t size) :
   fType(type),
   fRule(Integration::kGAUSS31),
   fAbsTol(absTol),
   fRelTol(relTol),
   fSize(size),
   fMaxIntervals(size),
   fResult(0),fError(0),fStatus(-1),fNEval(-1),
   fFunction(0),
   fWorkspace(0)
{

   // constructor with default rule (gauss31) passing the type
   // allocate workspace (only if not adaptive algorithm)
   if (type !=  Integration::kNONADAPTIVE)
      fWorkspace = new GSLIntegrationWorkspace( fSize);

}

   GSLIntegrator::GSLIntegrator(const char * type , int rule, double absTol, double relTol, size_t size) :
   fRule(Integration::kGAUSS31),
   fAbsTol(absTol),
   fRelTol(relTol),
   fSize(size),
   fMaxIntervals(size),
   fResult(0),fError(0),fStatus(-1),fNEval(-1),
   fFunction(0),
   fWorkspace(0)
{
   //std::cout << type << std::endl;

   fType =  Integration::kADAPTIVESINGULAR;  // default
   if (type != 0) {  // use this dafault
      std::string typeName(type);
      std::transform(typeName.begin(), typeName.end(), typeName.begin(), (int(*)(int)) toupper );
      if (typeName == "NONADAPTIVE")
         fType =  Integration::kNONADAPTIVE;
      else if (typeName == "ADAPTIVE")
         fType =  Integration::kADAPTIVE;
      else {
         if (typeName != "ADAPTIVESINGULAR")
            MATH_WARN_MSG("GSLIntegrator","Use default type: AdaptiveSingular");
      }
   }


   // constructor with default rule (gauss31) passing the type
   // allocate workspace (only if not adaptive algorithm)
   if (fType !=  Integration::kNONADAPTIVE)
      fWorkspace = new GSLIntegrationWorkspace( fSize);

   if (rule >= Integration::kGAUSS15 && rule <= Integration::kGAUSS61) SetIntegrationRule((Integration::GKRule) rule);

}


GSLIntegrator::~GSLIntegrator()
{
   // delete workspace and function
   if (fFunction) delete fFunction;
   if (fWorkspace) delete fWorkspace;
}

GSLIntegrator::GSLIntegrator(const GSLIntegrator &)  :
   VirtualIntegratorOneDim()
{
   // dummy copy ctr
}

GSLIntegrator & GSLIntegrator::operator = (const GSLIntegrator &rhs)
{
   // dummy operator=
   if (this == &rhs) return *this;  // time saving self-test

   return *this;
}




void  GSLIntegrator::SetFunction( GSLFuncPointer  fp, void * p) {
   // fill GSLFunctionWrapper with the pointer to the function
   if (fFunction ==0) fFunction = new GSLFunctionWrapper();
   fFunction->SetFuncPointer( fp );
   fFunction->SetParams ( p );
}

void  GSLIntegrator::SetFunction(const IGenFunction &f ) {
   // set function (make a copy of it)
   if (fFunction ==0) fFunction = new GSLFunctionWrapper();
   fFunction->SetFunction(f);
}

// evaluation methods

double  GSLIntegrator::Integral(double a, double b) {
   // defined integral evaluation
   // need here look at all types of algorithms
   // find more elegant solution ? Use template OK, but need to chose algorithm statically , t.b.i.

   if (!CheckFunction()) return 0;

   if ( fType == Integration::kNONADAPTIVE) {
      size_t neval = 0; // need to export  this ?
      fStatus = gsl_integration_qng( fFunction->GetFunc(), a, b , fAbsTol, fRelTol, &fResult, &fError, &neval);
      fNEval = neval;
   }
   else if (fType ==  Integration::kADAPTIVE) {
      fStatus = gsl_integration_qag( fFunction->GetFunc(), a, b , fAbsTol, fRelTol, fMaxIntervals, fRule, fWorkspace->GetWS(), &fResult, &fError);
      const int npts[6] = {15,21,31,41,51,61};
      assert(fRule>=1 && fRule <=6);
      fNEval = (fWorkspace->GetWS()->size)*npts[fRule-1];   // get size of workspace (number of iterations)
   }
   else if (fType ==  Integration::kADAPTIVESINGULAR) {

      // singular integration - look if we know about singular points


      fStatus = gsl_integration_qags( fFunction->GetFunc(), a, b , fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
      fNEval = (fWorkspace->GetWS()->size) * 21; //since 21 point rule is used in qags
   }
   else {

      fResult = 0;
      fError = 0;
      fStatus = -1;
      std::cerr << "GSLIntegrator - Error: Unknown integration type" << std::endl;
      throw std::exception(); //"Unknown integration type");
   }

   return fResult;

}

//=============================
double  GSLIntegrator::IntegralCauchy(double a, double b, double c) {
   //eval integral with Cauchy principal value defined at the value c
   if (!CheckFunction()) return 0;

   fStatus = gsl_integration_qawc( fFunction->GetFunc(), a, b , c, fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   fNEval = (fWorkspace->GetWS()->size) * 15; // 15 point rule is used ?

   return fResult;

}

double  GSLIntegrator::IntegralCauchy(const IGenFunction & f, double a, double b, double c) {
   //eval integral with Cauchy principal value defined at the value c

   if (!CheckFunction()) return 0;
   SetFunction(f);
   return IntegralCauchy(a, b, c);

}

//==============================

double  GSLIntegrator::Integral( const std::vector<double> & pts) {
   // integral eval with singularities

   if (!CheckFunction()) return 0;

   if (fType == Integration::kADAPTIVESINGULAR && pts.size() >= 2 ) {
      // remove constness ( should be const in GSL ? )
      double * p = const_cast<double *>(&pts.front() );
      fStatus = gsl_integration_qagp( fFunction->GetFunc(), p, pts.size() , fAbsTol, fRelTol, fMaxIntervals,  fWorkspace->GetWS(), &fResult, &fError);
      fNEval = (fWorkspace->GetWS()->size) * 15; // 15 point rule is used ?
   }
   else {
      fResult = 0;
      fError = 0;
      fStatus = -1;
      std::cerr << "GSLIntegrator - Error: Unknown integration type or not enough singular points defined" << std::endl;
      return 0;
   }
   return fResult;
}


double  GSLIntegrator::Integral( ) {
   // Eval for indefined integrals: use QAGI method
   // if method was chosen NO_ADAPTIVE WS does not exist create it

   if (!CheckFunction()) return 0;

   if (!fWorkspace) fWorkspace = new GSLIntegrationWorkspace( fSize);

   fStatus = gsl_integration_qagi( fFunction->GetFunc(), fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   fNEval = (fWorkspace->GetWS()->size) * 15; // 15 point rule is used ?

   return fResult;
}



double  GSLIntegrator::IntegralUp( double a ) {
   // Integral between [a, + inf]
   // if method was chosen NO_ADAPTIVE WS does not exist create it

   if (!CheckFunction()) return 0;

   if (!fWorkspace) fWorkspace = new GSLIntegrationWorkspace( fSize);

   fStatus = gsl_integration_qagiu( fFunction->GetFunc(), a, fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   fNEval = (fWorkspace->GetWS()->size) * 21; // 21 point rule is used ?

   return fResult;
}



double  GSLIntegrator::IntegralLow( double b ) {
   // Integral between [-inf, + b]
   // if method was chosen NO_ADAPTIVE WS does not exist create it

   if (!CheckFunction()) return 0;

   if (!fWorkspace) fWorkspace = new GSLIntegrationWorkspace( fSize);

   fStatus = gsl_integration_qagil( fFunction->GetFunc(), b, fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   fNEval = (fWorkspace->GetWS()->size) * 21; // 21 point rule is used ?

   return fResult;
}




double  GSLIntegrator::Integral(const IGenFunction & f, double a, double b) {
   // use generic function interface
   SetFunction(f);
   return Integral(a,b);
}

double  GSLIntegrator::Integral(const IGenFunction & f ) {
   // use generic function interface
   SetFunction(f);
   return Integral();
}

double  GSLIntegrator::IntegralUp(const IGenFunction & f, double a) {
   // use generic function interface
   SetFunction(f);
   return IntegralUp(a);
}

double  GSLIntegrator::IntegralLow(const IGenFunction & f, double b) {
   // use generic function interface
   SetFunction(f);
   return IntegralLow(b);
}

double  GSLIntegrator::Integral(const IGenFunction & f, const std::vector<double> & pts) {
   // use generic function interface
   SetFunction(f);
   return Integral(pts);
}




double  GSLIntegrator::Integral( GSLFuncPointer f , void * p, double a, double b) {
   // use c free function pointer
   SetFunction( f, p);
   return Integral(a,b);
}

double  GSLIntegrator::Integral( GSLFuncPointer f, void * p ) {
   // use c free function pointer
   SetFunction( f, p);
   return Integral();
}

double  GSLIntegrator::IntegralUp( GSLFuncPointer f, void * p, double a ) {
   // use c free function pointer
   SetFunction( f, p);
   return IntegralUp(a);
}

double  GSLIntegrator::IntegralLow( GSLFuncPointer f, void * p, double b ) {
   // use c free function pointer
   SetFunction( f, p);
   return IntegralLow(b);
}

double  GSLIntegrator::Integral( GSLFuncPointer f, void * p, const std::vector<double> & pts ) {
   // use c free function pointer
   SetFunction( f, p);
   return Integral(pts);
}



double GSLIntegrator::Result() const { return fResult; }

double GSLIntegrator::Error() const { return fError; }

int GSLIntegrator::Status() const { return fStatus; }


// get and setter methods

//   double GSLIntegrator::getAbsTolerance() const { return fAbsTol; }

void GSLIntegrator::SetAbsTolerance(double absTol){ this->fAbsTol = absTol; }

//   double GSLIntegrator::getRelTolerance() const { return fRelTol; }

void GSLIntegrator::SetRelTolerance(double relTol){ this->fRelTol = relTol; }


void GSLIntegrator::SetIntegrationRule(Integration::GKRule rule){ this->fRule = rule; }

bool GSLIntegrator::CheckFunction() {
   // check if a function has been previously set.
   if (fFunction->IsValid()) return true;
   fStatus = -1; fResult = 0; fError = 0;
   std::cerr << "GSLIntegrator - Error : Function has not been specified " << std::endl;
   return false;
}

void GSLIntegrator::SetOptions(const ROOT::Math::IntegratorOneDimOptions & opt)
{
   //   set integration options
   fType = opt.IntegratorType();
   if (fType == IntegrationOneDim::kDEFAULT) fType = IntegrationOneDim::kADAPTIVESINGULAR;
   if (fType != IntegrationOneDim::kADAPTIVE &&
       fType != IntegrationOneDim::kADAPTIVESINGULAR &&
       fType != IntegrationOneDim::kNONADAPTIVE ) {
      MATH_WARN_MSG("GSLIntegrator::SetOptions","Invalid rule options - use default ADAPTIVESINGULAR");
      fType = IntegrationOneDim::kADAPTIVESINGULAR;
   }
   SetAbsTolerance( opt.AbsTolerance() );
   SetRelTolerance( opt.RelTolerance() );
   fSize = opt.WKSize();
   fMaxIntervals = fSize;
   if (fType == Integration::kADAPTIVE) {
      int npts = opt.NPoints();
      if  ( npts >= Integration::kGAUSS15 && npts <= Integration::kGAUSS61)
         fRule = (Integration::GKRule) npts;
      else {
         MATH_WARN_MSG("GSLIntegrator::SetOptions","Invalid rule options - use default GAUSS31");
         fRule = Integration::kGAUSS31;
      }
   }
}

ROOT::Math::IntegratorOneDimOptions  GSLIntegrator::Options() const {
   ROOT::Math::IntegratorOneDimOptions opt;
   opt.SetAbsTolerance(fAbsTol);
   opt.SetRelTolerance(fRelTol);
   opt.SetWKSize(fSize);
   opt.SetIntegrator(GetTypeName() );

   if (fType == IntegrationOneDim::kADAPTIVE)
      opt.SetNPoints(fRule);
   else if (fType == IntegrationOneDim::kADAPTIVESINGULAR)
      opt.SetNPoints( Integration::kGAUSS31 ); // fixed rule for adaptive singular
   else
      opt.SetNPoints( 0 ); // not available for the rest

   return opt;
}

const char * GSLIntegrator::GetTypeName() const {
   if (fType == IntegrationOneDim::kADAPTIVE) return "Adaptive";
   if (fType == IntegrationOneDim::kADAPTIVESINGULAR) return "AdaptiveSingular";
   if (fType == IntegrationOneDim::kNONADAPTIVE) return "NonAdaptive";
   return "Undefined";
}


} // namespace Math
} // namespace ROOT
