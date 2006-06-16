// @(#)root/mathmore:$Name:  $:$Id: Integrator.cxx,v 1.3 2005/09/18 20:41:25 brun Exp $
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

#include "Math/Integrator.h"
#include "GSLIntegrator.h"

//#include "GSLIntegrationWorkspace.h"
//#include "MathMore/GSLFunctionWrapper.h"

#include "gsl/gsl_integration.h"



namespace ROOT {
namespace Math {



  

Integrator::Integrator(const IGenFunction &f, Integration::Type type , Integration::GKRule rule, double absTol, double relTol, size_t size) {
   // full constructor with IGenFunction
   fIntegrator = new GSLIntegrator(f, type, rule, absTol, relTol, size);
}


Integrator::Integrator(const IGenFunction &f, double absTol, double relTol, size_t size) {
   // constructor with default type (ADaptiveSingular) ,  rule is not needed  
   fIntegrator = new GSLIntegrator(f, absTol, relTol, size);
}



Integrator::Integrator(const IGenFunction &f, Integration::Type type , double absTol, double relTol, size_t size) {
   // constructor with default rule (gauss31) passing the type
   fIntegrator = new GSLIntegrator(f, type, absTol, relTol, size);
}

Integrator::Integrator(GSLFuncPointer f, Integration::Type type , Integration::GKRule rule, double absTol, double relTol, size_t size) {
   // full constructor with GSL function pointer type
   fIntegrator = new GSLIntegrator(f, type, rule, absTol, relTol, size);
}



Integrator::Integrator(GSLFuncPointer f, double absTol, double relTol, size_t size) {
   // constructor with GSL function pointer type and with default type (ADaptiveSingular) ,  rule is not needed  
   fIntegrator = new GSLIntegrator(f, absTol, relTol, size);
}



Integrator::Integrator(GSLFuncPointer f, Integration::Type type , double absTol, double relTol, size_t size) {
   // constructor with GSL function pointer type and with default rule (gauss31) passing the type
   fIntegrator = new GSLIntegrator(f, type, absTol, relTol, size);
}

Integrator::~Integrator()
{
   // destructor 
   if (fIntegrator) delete fIntegrator;
}

Integrator::Integrator(const Integrator &)
{
}

Integrator & Integrator::operator = (const Integrator &rhs)
{
   // private assigment op.
   if (this == &rhs) return *this;  // time saving self-test
   
   return *this;
}


void Integrator::SetFunction(const IGenFunction &f) {
   // set with IGenFunction type
   fIntegrator->SetFunction(f);
}

void Integrator::SetFunction( const GSLFuncPointer &f) {
   // set with GSL function type
   fIntegrator->SetFunction(f);
}




// evaluation methods

double  Integrator::Integral(double a, double b) {
   // evaluation defined integral
   return fIntegrator->Integral(a, b);
}


double  Integrator::Integral( const std::vector<double> & pts) {
   // evaluation singular integral
   return fIntegrator->Integral(pts);
}




double  Integrator::Integral( ) {
   // Eval for indefined integrals: use QAGI method
   return fIntegrator->Integral();
}



double  Integrator::IntegralUp( double a ) {
   // Integral between [a, + inf]
   return fIntegrator->IntegralUp(a);
}



double  Integrator::IntegralLow( double b ) {
   // Integral between [-inf, + b]
   return fIntegrator->IntegralLow(b);
}


// evaluations passing the function

double  Integrator::Integral(const IGenFunction & f, double a, double b) {
   // evaluation using generic function interface   
   return fIntegrator->Integral(f, a, b);
}

double  Integrator::Integral(const IGenFunction & f ) {
   // evaluation using generic function interface 
   return fIntegrator->Integral(f);
}

double  Integrator::IntegralUp(const IGenFunction & f, double a) {
   // evaluation using generic function interface    
   return fIntegrator->IntegralUp(f, a);
}

double  Integrator::IntegralLow(const IGenFunction & f, double b) {
   // evaluation using generic function interface 
   return fIntegrator->IntegralLow(f, b);
}

double  Integrator::Integral(const IGenFunction & f, const std::vector<double> & pts) {
   // evaluation using generic function interface 
   return fIntegrator->Integral(f, pts);
}



// use c free function pointer
double  Integrator::Integral( GSLFuncPointer f , void * p, double a, double b) {
   // eval using GSL function pointer type
   return fIntegrator->Integral(f, p, a, b);
}

double  Integrator::Integral( GSLFuncPointer f, void * p ) {
   // eval using GSL function pointer type
   return fIntegrator->Integral(f, p);
}

double  Integrator::IntegralUp( GSLFuncPointer f, void * p, double a ) {
   // eval using GSL function pointer type
   return fIntegrator->IntegralUp(f, p, a);
}

double  Integrator::IntegralLow( GSLFuncPointer f, void * p, double b ) {
   // eval using GSL function pointer type
   return fIntegrator->IntegralLow(f, p, b);
}

double  Integrator::Integral( GSLFuncPointer f, void * p, const std::vector<double> & pts ) {
   // eval using GSL function pointer type
   return fIntegrator->Integral(f, p, pts);
}



double Integrator::Result() const { return fIntegrator->Result(); }

double Integrator::Error() const { return fIntegrator->Error(); }

int Integrator::Status() const { return fIntegrator->Status(); }


// get and setter methods

//   double Integrator::getAbsTolerance() const { return fAbsTol; }

void Integrator::SetAbsTolerance(double absTol){
   // abs tolerance setter
   fIntegrator->SetAbsTolerance(absTol);
}

//   double Integrator::getRelTolerance() const { return fRelTol; }

void Integrator::SetRelTolerance(double relTol){
    // rel tolerance setter
   fIntegrator->SetRelTolerance(relTol);
}


void Integrator::SetIntegrationRule(Integration::GKRule rule){
   // set rule
   fIntegrator->SetIntegrationRule(rule);
}


} // namespace Math
} // namespace ROOT
