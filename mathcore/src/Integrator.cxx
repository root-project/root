// @(#)root/mathmore:$Id: Integrator.cxx 19826 2007-09-19 19:56:11Z rdm $
// Authors: L. Moneta, M. Slawinska 10/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

#include "Math/IFunction.h"
#include "Math/Integrator.h"

#include "Math/VirtualIntegrator.h"
#include "Math/IntegratorMultiDim.h"


#define MATH_PLUGIN_MANAGER
#ifdef MATH_PLUGIN_MANAGER
#include "TROOT.h"
#include "TPluginManager.h"
#else
#include "Math/GSLIntegrator.h"
#include "Math/GSLMCIntegrator.h"
#endif

#include <cassert>

namespace ROOT {
namespace Math {


VirtualIntegrator * CreateOneDimIntegrator(IntegrationOneDim::Type type , double absTol, double relTol, unsigned int size) { 

   VirtualIntegrator * ig = 0; 

#ifdef MATH_PLUGIN_MANAGER  
   TPluginHandler *h; 
   //gDebug = 3; 
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::VirtualIntegrator", "GSLIntegrator"))) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("Error loading one dimensional GSL integrator"); 
         return 0; 
      }

      std::string typeName = "ADAPTIVE";
      if (type == IntegrationOneDim::ADAPTIVESINGULAR) 
         typeName = "ADAPTIVESINGULAR";
      if (type == IntegrationOneDim::NONADAPTIVE) 
         typeName = "NONADAPTIVE";

            

      ig = reinterpret_cast<ROOT::Math::VirtualIntegrator *>( h->ExecPlugin(4,typeName.c_str(), absTol, relTol, size ) ); 
      assert(ig != 0);

#ifdef DEBUG
      std::cout << "Loaded Integrator " << typeid(*ig).name() << std::endl;
#endif
   }
#else 
   ig =  new GSLIntegrator(type, absTol, relTol, size);
#endif
   return ig;  
}

VirtualIntegrator * CreateMultiDimIntegrator(IntegrationMultiDim::Type type , double absTol, double relTol, unsigned int ncall) { 

   VirtualIntegrator * ig = 0; 
   
#ifdef MATH_PLUGIN_MANAGER  

   if (type == IntegrationMultiDim::ADAPTIVE) { 
      // no need of PM for adaptive method (is in mathcore)
      return new IntegratorMultiDim(absTol, relTol, ncall);
   }



   TPluginHandler *h; 
   //gDebug = 3; 
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::VirtualIntegrator", "GSLMCIntegrator"))) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("Error loading multidim integrator"); 
         return 0; 
      }


      std::string typeName = "VEGAS";
      if (type == IntegrationMultiDim::MISER) 
         typeName = "MISER";
      if (type == IntegrationMultiDim::PLAIN) 
         typeName = "PLAIN";

      
      ig = reinterpret_cast<ROOT::Math::VirtualIntegrator *>( h->ExecPlugin(4,typeName.c_str(), absTol, relTol, ncall ) ); 
      assert(ig != 0);

#ifdef DEBUG
      std::cout << "Loaded Integrator " << typeid(*ig).name() << std::endl;
#endif
   }
#else 
   if (type == IntegrationMultiDim::ADAPTIVE)
      ig = new IntegratorMultiDim(absTol, relTol, ncall);
   else {
      // do later 
      //ig =  new GSLMCIntegrator(type, absTol, relTol, ncall);
   }
#endif
   return ig;  
}

// 1D constructors

Integrator::Integrator(IntegrationOneDim::Type type , double absTol, double relTol, unsigned int size)  
{
   // construtor of 1D integrator - create a GSLIntegrator
   fIntegrator = CreateOneDimIntegrator(type, absTol, relTol, size); 
}
  

Integrator::Integrator(const IGenFunction &f, IntegrationOneDim::Type type , double absTol, double relTol, unsigned int size) {
   // constructor with IGenFunction
   fIntegrator = CreateOneDimIntegrator(type, absTol, relTol, size); 
   fIntegrator->SetFunction(f);
}

// multi-dim constructors

Integrator::Integrator(IntegrationMultiDim::Type type , double absTol, double relTol, unsigned int ncall)  
{
   // construtor of multi-dima integrator 
   fIntegrator = CreateMultiDimIntegrator(type, absTol, relTol, ncall); 
}

Integrator::Integrator(const IMultiGenFunction &f, IntegrationMultiDim::Type type , double absTol, double relTol, unsigned int ncall)  
{
   // construtor of multi-dim integrator using a funciton interface
   fIntegrator = CreateMultiDimIntegrator(type, absTol, relTol, ncall); 
   fIntegrator->SetFunction(f);   
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



} // namespace Math
} // namespace ROOT
