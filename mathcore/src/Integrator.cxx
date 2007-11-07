// @(#)root/mathmore:$Id: Integrator.cxx 19826 2007-09-19 19:56:11Z rdm $
// Authors: L. Moneta, M. Slawinska 10/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

#include "Math/IFunction.h"
#include "Math/VirtualIntegrator.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"

#include "Math/AdaptiveIntegratorMultiDim.h"



#ifndef MATH_NO_PLUGIN_MANAGER
#include "TROOT.h"
#include "TPluginManager.h"

#else // case no plugin manager is available
#include "Math/GSLIntegrator.h"
#include "Math/GSLMCIntegrator.h"
#endif

#include <cassert>

namespace ROOT {
namespace Math {


   VirtualIntegratorOneDim * IntegratorOneDim::CreateIntegrator(IntegrationOneDim::Type type , double absTol, double relTol, unsigned int size, int rule) { 

   VirtualIntegratorOneDim * ig = 0; 

#ifdef MATH_NO_PLUGIN_MANAGER    // no PM available
   ig =  new GSLIntegrator(type, absTol, relTol, size);
#else 

   TPluginHandler *h; 
   //gDebug = 3; 
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::VirtualIntegrator", "GSLIntegrator"))) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("IntegratorOneDim::CreateIntegrator","Error loading one dimensional GSL integrator"); 
         return 0; 
      }

      std::string typeName = "ADAPTIVE";
      if (type == IntegrationOneDim::ADAPTIVESINGULAR) 
         typeName = "ADAPTIVESINGULAR";
      if (type == IntegrationOneDim::NONADAPTIVE) 
         typeName = "NONADAPTIVE";

            

      ig = reinterpret_cast<ROOT::Math::VirtualIntegratorOneDim *>( h->ExecPlugin(5,typeName.c_str(), rule, absTol, relTol, size ) ); 
      assert(ig != 0);
      

#ifdef DEBUG
      std::cout << "Loaded Integrator " << typeid(*ig).name() << std::endl;
#endif
   }

#endif 

   return ig;  
}

VirtualIntegratorMultiDim * IntegratorMultiDim::CreateIntegrator(IntegrationMultiDim::Type type , double absTol, double relTol, unsigned int ncall) { 
   // create base class for multidimensional integration 

   // no need for PM in the adaptive  case using Genz method (class is in MathCore)
   if (type == IntegrationMultiDim::ADAPTIVE)
      return new AdaptiveIntegratorMultiDim(absTol, relTol, ncall);
      
   VirtualIntegratorMultiDim * ig = 0; 

#ifdef MATH_NO_PLUGIN_MANAGER  // no PM available 
   ig =  new GSLMCIntegrator(type, absTol, relTol, ncall);

#else  // use ROOT PM 
      
   TPluginHandler *h; 
   //gDebug = 3; 
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::VirtualIntegrator", "GSLMCIntegrator"))) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("IntegratorMultiDim::CreateIntegrator","Error loading multidim integrator"); 
         return 0; 
      }


      std::string typeName = "VEGAS";
      if (type == IntegrationMultiDim::MISER) 
         typeName = "MISER";
      if (type == IntegrationMultiDim::PLAIN) 
         typeName = "PLAIN";

      
      ig = reinterpret_cast<ROOT::Math::VirtualIntegratorMultiDim *>( h->ExecPlugin(4,typeName.c_str(), absTol, relTol, ncall ) ); 
      assert(ig != 0);

#ifdef DEBUG
      std::cout << "Loaded Integrator " << typeid(*ig).name() << std::endl;
#endif
   }
#endif
   return ig;  
}




} // namespace Math
} // namespace ROOT
