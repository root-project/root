// @(#)root/mathmore:$Name:  $:$Id: GSLIntegrator.cxx,v 1.5 2006/06/19 08:44:08 moneta Exp $
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

#include "gsl/gsl_integration.h"
#include "GSLIntegrator.h"
#include "Math/IFunction.h"


#include "GSLIntegrationWorkspace.h"
#include "GSLFunctionWrapper.h"




namespace ROOT {
namespace Math {


 

GSLIntegrator::GSLIntegrator(const IGenFunction &f, const Integration::Type type , const Integration::GKRule rule, double absTol, double relTol, size_t size) :
   fType(type),
   fRule(rule),
   fAbsTol(absTol),
   fRelTol(relTol),
   fSize(size),
   fMaxIntervals(size),
   fWorkspace(0),
   fFunction(0)
{
   // full constructor 
   // allocate workspace (only if not adaptive algorithm)
   if (type !=  Integration::NONADAPTIVE)
   fWorkspace = new GSLIntegrationWorkspace( fSize);
      
   fFunction = new GSLFunctionWrapper();
   SetFunction(f);
      
}



GSLIntegrator::GSLIntegrator(const IGenFunction &f, double absTol, double relTol, size_t size) :
   fType(Integration::ADAPTIVESINGULAR),
   fRule(Integration::GAUSS31),
   fAbsTol(absTol),
   fRelTol(relTol),
   fSize(size),
   fMaxIntervals(size),
   fWorkspace(0),
   fFunction(0)
{
   // constructor with default type (ADaptiveSingular) ,  rule is not needed  
   fWorkspace = new GSLIntegrationWorkspace( fSize);
   fFunction = new GSLFunctionWrapper();
   SetFunction(f);
   
}



GSLIntegrator::GSLIntegrator(const IGenFunction &f, const Integration::Type type , double absTol, double relTol, size_t size) :
fType(type),
fRule(Integration::GAUSS31),
fAbsTol(absTol),
fRelTol(relTol),
fSize(size),
fMaxIntervals(size),
fWorkspace(0),
fFunction(0)
{
   // constructor with default rule (gauss31) passing the type
   // allocate workspace (only if not adaptive algorithm)
   if (type !=  Integration::NONADAPTIVE)
      fWorkspace = new GSLIntegrationWorkspace( fSize);
   
   fFunction = new GSLFunctionWrapper();
   SetFunction(f);
}

GSLIntegrator::GSLIntegrator(GSLFuncPointer f, const Integration::Type type , const Integration::GKRule rule, double absTol, double relTol, size_t size) :
fType(type),
fRule(rule),
fAbsTol(absTol),
fRelTol(relTol),
fSize(size),
fMaxIntervals(size),
fWorkspace(0),
fFunction(0)
{
   // constructor
   // allocate workspace (only if not adaptive algorithm)
   if (type !=  Integration::NONADAPTIVE)
      fWorkspace = new GSLIntegrationWorkspace( fSize);
   
   fFunction = new GSLFunctionWrapper();
   SetFunction(f);
   
}



GSLIntegrator::GSLIntegrator(GSLFuncPointer f, double absTol, double relTol, size_t size) :
fType(Integration::ADAPTIVESINGULAR),
fRule(Integration::GAUSS31),
fAbsTol(absTol),
fRelTol(relTol),
fSize(size),
fMaxIntervals(size),
fWorkspace(0),
fFunction(0)
{
   // constructor with default type (ADaptiveSingular) ,  rule is not needed
   fWorkspace = new GSLIntegrationWorkspace( fSize);
   fFunction = new GSLFunctionWrapper();
   SetFunction(f);
}



GSLIntegrator::GSLIntegrator(GSLFuncPointer f, const Integration::Type type , double absTol, double relTol, size_t size) :
fType(type),
fRule(Integration::GAUSS31),
fAbsTol(absTol),
fRelTol(relTol),
fSize(size),
fMaxIntervals(size),
fWorkspace(0),
fFunction(0)
{
   // constructor with default rule (gauss31) passing the type
   // allocate workspace (only if not adaptive algorithm)
   if (type !=  Integration::NONADAPTIVE)
      fWorkspace = new GSLIntegrationWorkspace( fSize);
   
   fFunction = new GSLFunctionWrapper();
   SetFunction(f);
}

GSLIntegrator::~GSLIntegrator()
{
   // delete workspace and function holders
   if (fWorkspace) delete fWorkspace;
   if (fFunction) delete fFunction;
}

GSLIntegrator::GSLIntegrator(const GSLIntegrator &)
{
   // dummy copy ctr
}

GSLIntegrator & GSLIntegrator::operator = (const GSLIntegrator &rhs)
{
   // dummy operator=
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}




void  GSLIntegrator::FillGSLFunction( GSLFuncPointer  fp, void * p) {
   // fill GSLFunctionWrapper with the pointer to the function 
   fFunction->SetFuncPointer( fp );
   fFunction->SetParams ( p );
}

void  GSLIntegrator::FillGSLFunction(const IGenFunction &f) {
   // set function
   fFunction->SetFunction(f);
}

// evaluation methods

double  GSLIntegrator::Integral(double a, double b) {
   // defined integral evaluation
   // need here look at all types of algorithms
   // find more elegant solution ? Use template OK, but need to chose algorithm statically , t.b.i.
   
   
   if ( fType == Integration::NONADAPTIVE) {
      size_t neval = 0; // need to export  this ?
      fStatus = gsl_integration_qng( fFunction->GetFunc(), a, b , fAbsTol, fRelTol, &fResult, &fError, &neval);
   }
   else if (fType ==  Integration::ADAPTIVE) {
      fStatus = gsl_integration_qag( fFunction->GetFunc(), a, b , fAbsTol, fRelTol, fMaxIntervals, fRule, fWorkspace->GetWS(), &fResult, &fError);
   }
   else if (fType ==  Integration::ADAPTIVESINGULAR) {
      
      // singular integration - look if we know about singular points
      
      
      fStatus = gsl_integration_qags( fFunction->GetFunc(), a, b , fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   }
   else {
      
      fResult = 0;
      fError = 0;
      fStatus = -1;
      throw std::exception(); //"Unknown integration type");
   }
   
   return fResult;
   
}


double  GSLIntegrator::Integral( const std::vector<double> & pts) {
   // integral eval with singularities
   if (fType == Integration::ADAPTIVESINGULAR && pts.size() >= 2 ) {
      // remove constness ( should be const in GSL ? )
      double * p = const_cast<double *>(&pts.front() );
      fStatus = gsl_integration_qagp( fFunction->GetFunc(), p, pts.size() , fAbsTol, fRelTol, fMaxIntervals,  fWorkspace->GetWS(), &fResult, &fError);
   }
   else {
      fResult = 0;
      fError = 0;
      fStatus = -1;
      throw std::exception(); //"Wrong integration type or no singular points defined");
   }
   return fResult;
}




double  GSLIntegrator::Integral( ) {
   // Eval for indefined integrals: use QAGI method
   // if method was choosen NO_ADAPTIVE WS does not exist create it
   if (!fWorkspace) fWorkspace = new GSLIntegrationWorkspace( fSize);
   
   fStatus = gsl_integration_qagi( fFunction->GetFunc(), fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   
   return fResult;
}



double  GSLIntegrator::IntegralUp( double a ) {
   // Integral between [a, + inf]
   // if method was choosen NO_ADAPTIVE WS does not exist create it
   if (!fWorkspace) fWorkspace = new GSLIntegrationWorkspace( fSize);
   
   fStatus = gsl_integration_qagiu( fFunction->GetFunc(), a, fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   
   return fResult;
}



double  GSLIntegrator::IntegralLow( double b ) {
   // Integral between [-inf, + b]
   // if method was choosen NO_ADAPTIVE WS does not exist create it
   if (!fWorkspace) fWorkspace = new GSLIntegrationWorkspace( fSize);
   
   fStatus = gsl_integration_qagil( fFunction->GetFunc(), b, fAbsTol, fRelTol, fMaxIntervals, fWorkspace->GetWS(), &fResult, &fError);
   
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
   FillGSLFunction( f, p);
   return Integral(a,b);
}

double  GSLIntegrator::Integral( GSLFuncPointer f, void * p ) {
   // use c free function pointer
   FillGSLFunction( f, p);
   return Integral();
}

double  GSLIntegrator::IntegralUp( GSLFuncPointer f, void * p, double a ) {
   // use c free function pointer
   FillGSLFunction( f, p);
   return IntegralUp(a);
}

double  GSLIntegrator::IntegralLow( GSLFuncPointer f, void * p, double b ) {
   // use c free function pointer
   FillGSLFunction( f, p);
   return IntegralLow(b);
}

double  GSLIntegrator::Integral( GSLFuncPointer f, void * p, const std::vector<double> & pts ) {
   // use c free function pointer
   FillGSLFunction( f, p);
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


} // namespace Math
} // namespace ROOT
