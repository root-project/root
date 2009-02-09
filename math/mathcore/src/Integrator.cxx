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

#include "Math/GaussIntegrator.h"

#include "Math/OneDimFunctionAdapter.h"


#include "RConfigure.h"
// #ifndef ROOTINCDIR
// #define MATH_NO_PLUGIN_MANAGER
// #endif

#ifndef MATH_NO_PLUGIN_MANAGER

#include "TROOT.h"
#include "TPluginManager.h"

#else // case no plugin manager is available
#ifdef R__HAS_MATHMORE
#include "Math/GSLIntegrator.h"
#include "Math/GSLMCIntegrator.h"
#endif

#endif

#include <cassert>

namespace ROOT {
namespace Math {

void IntegratorOneDim::SetFunction(const IMultiGenFunction &f, unsigned int icoord , const double * x ) { 
   // set function from a multi-dim function 
   // pass also x in case of multi-dim function express the other dimensions (which are fixed) 
   unsigned int ndim = f.NDim(); 
   assert (icoord < ndim); 
   ROOT::Math::OneDimMultiFunctionAdapter<> adapter(f,ndim,icoord);
   // case I pass a vector x which is needed (for example to compute I(y) = Integral( f(x,y) dx) ) need to setCX
   if (x != 0) adapter.SetX(x, x+ ndim);
   SetFunction(adapter,true); // need to copy this object
}


// methods to create integrators 

VirtualIntegratorOneDim * IntegratorOneDim::CreateIntegrator(IntegrationOneDim::Type type , double absTol, double relTol, unsigned int size, int rule) { 
   // create the concrete class for one-dimensional integration. Use the plug-in manager if needed 

#ifndef R__HAS_MATHMORE   
   // default type is GAUSS when Mathmore is not built
   type = IntegrationOneDim::kGAUSS; 
#endif


   if (type == IntegrationOneDim::kGAUSS)
      return new GaussIntegrator();

   VirtualIntegratorOneDim * ig = 0; 

#ifdef MATH_NO_PLUGIN_MANAGER    // no PM available
#ifdef R__HAS_MATHMORE   
   ig =  new GSLIntegrator(type, absTol, relTol, size);
#else 
   MATH_ERROR_MSG("IntegratorOneDim::CreateIntegrator","Integrator type is not available in MathCore");
#endif

#else  // case of using Plugin Manager
   


   TPluginHandler *h; 
   //gDebug = 3; 
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::VirtualIntegrator", "GSLIntegrator"))) {
      if (h->LoadPlugin() == -1) {
         MATH_WARN_MSG("IntegratorOneDim::CreateIntegrator","Error loading one dimensional GSL integrator - use Gauss integrator"); 
         return new GaussIntegrator();
      }

      std::string typeName = "ADAPTIVE";
      if (type == IntegrationOneDim::kADAPTIVESINGULAR) 
         typeName = "ADAPTIVESINGULAR";
      if (type == IntegrationOneDim::kNONADAPTIVE) 
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
   // create concrete class for multidimensional integration 

#ifndef R__HAS_MATHMORE   
   // default type is Adaptive when Mathmore is not built
   type = IntegrationMultiDim::kADAPTIVE; 
#endif

   // no need for PM in the adaptive  case using Genz method (class is in MathCore)
   if (type == IntegrationMultiDim::kADAPTIVE)
      return new AdaptiveIntegratorMultiDim(absTol, relTol, ncall);
      
   VirtualIntegratorMultiDim * ig = 0; 

#ifdef MATH_NO_PLUGIN_MANAGER  // no PM available 
#ifdef R__HAS_MATHMORE   
   ig =  new GSLMCIntegrator(type, absTol, relTol, ncall);
#else 
   MATH_ERROR_MSG("IntegratorMultiDim::CreateIntegrator","Integrator type is not available in MathCore");
#endif

#else  // use ROOT PM 
      
   TPluginHandler *h; 
   //gDebug = 3; 
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::VirtualIntegrator", "GSLMCIntegrator"))) {
      if (h->LoadPlugin() == -1) {
         MATH_WARN_MSG("IntegratorMultiDim::CreateIntegrator","Error loading GSL MC multidim integrator - use adaptive method"); 
         return new AdaptiveIntegratorMultiDim(absTol, relTol, ncall);
      }


      std::string typeName = "VEGAS";
      if (type == IntegrationMultiDim::kMISER) 
         typeName = "MISER";
      if (type == IntegrationMultiDim::kPLAIN) 
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
