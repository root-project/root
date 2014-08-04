// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Dec 22 14:43:33 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class MinimizerFactory

#include "Math/Factory.h"
#include "Math/Error.h"

#include "RConfigure.h"

#include "Math/Minimizer.h"
#include "Math/MinimizerOptions.h"

#include "Math/DistSampler.h"
#include "Math/DistSamplerOptions.h"

// uncomment these if you dont want to use the plugin manager
// but you need to link also  the needed minimization libraries (e.g Minuit and/or Minuit2)
// #define MATH_NO_PLUGIN_MANAGER
// #define HAS_MINUIT
// #define HAS_MINUIT2

#ifndef MATH_NO_PLUGIN_MANAGER
// use ROOT Plug-in manager
#include "TPluginManager.h"
#include "TROOT.h"
#include "TVirtualMutex.h"
#else
// all the minimizer implementation classes
//#define HAS_MINUIT2
#ifdef HAS_MINUIT2
#include "Minuit2/Minuit2Minimizer.h"
#endif
#ifdef HAS_MINUIT
#include "TMinuitMinimizer.h"
#endif
#ifdef R__HAS_MATHMORE
#include "Math/GSLMinimizer.h"
#include "Math/GSLNLSMinimizer.h"
#include "Math/GSLSimAnMinimizer.h"
#endif

#endif

#include <algorithm>
#include <cassert>

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

#ifndef MATH_NO_PLUGIN_MANAGER
// use ROOT Plugin Manager to create Minimizer concrete classes

ROOT::Math::Minimizer * ROOT::Math::Factory::CreateMinimizer(const std::string & minimizerType,const std::string & algoType)
{
   // create Minimizer using the plug-in manager given the type of Minimizer (MINUIT, MINUIT2, FUMILI, etc..) and
   // algorithm (MIGRAD, SIMPLEX, etc..)

   const char * minim = minimizerType.c_str();
   const char * algo = algoType.c_str();

   //case of fumili2
   std::string s1,s2;
   if (minimizerType == "Fumili2" ) {
      s1 = "Minuit2";
      s2 = "fumili";
      minim = s1.c_str();
      algo =  s2.c_str();
   }
   if (minimizerType == "TMinuit") {
      s1 = "Minuit";
      minim = s1.c_str();
   }

   if (minimizerType.empty() ) minim = ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();

   R__LOCKGUARD2(gROOTMutex);

   // create Minimizer using the PM
   TPluginHandler *h;
   //gDebug = 3;
   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::Minimizer",minim ))) {
      if (h->LoadPlugin() == -1)  {
#ifdef DEBUG
      std::cout << "Error Loading ROOT::Math::Minimizer " << minim << std::endl;
#endif
         return 0;
      }

      // create plug-in with required algorithm
      ROOT::Math::Minimizer * min = reinterpret_cast<ROOT::Math::Minimizer *>( h->ExecPlugin(1,algo ) );
#ifdef DEBUG
      if (min != 0)
         std::cout << "Loaded Minimizer " << minimizerType << "  " << algoType << std::endl;
      else
         std::cout << "Error creating Minimizer " << minimizerType << "  " << algoType << std::endl;
#endif

      return min;
   }
   return 0;

}

#else

// use directly classes instances

ROOT::Math::Minimizer * ROOT::Math::Factory::CreateMinimizer(const std::string & minimizerType, const std::string & algoType)
{
   // static method to create a minimizer .
   // not using PM so direct dependency on all libraries (Minuit, Minuit2, MathMore, etc...)
   // The default is the Minuit2 minimizer or GSL Minimizer

   // should use enumerations instead of string ?

   Minimizer * min = 0;
   std::string algo = algoType;


#ifdef HAS_MINUIT2
   if (minimizerType ==  "Minuit2")
      min = new ROOT::Minuit2::Minuit2Minimizer(algoType.c_str());
   if (minimizerType ==  "Fumili2")
      min = new ROOT::Minuit2::Minuit2Minimizer("fumili");
#endif

#ifdef HAS_MINUIT
   // use TMinuit
   if (minimizerType ==  "Minuit" || minimizerType ==  "TMinuit")
      min = new TMinuitMinimizer(algoType.c_str());
#endif

#ifdef R__HAS_MATHMORE
   // use GSL minimizer
   if (minimizerType ==  "GSL")
      min = new ROOT::Math::GSLMinimizer(algoType.c_str());

   else if (minimizerType ==  "GSL_NLS")
      min = new ROOT::Math::GSLNLSMinimizer();

   else if (minimizerType ==  "GSL_SIMAN")
      min = new ROOT::Math::GSLSimAnMinimizer();
#endif


#ifdef HAS_MINUIT2
   // DEFAULT IS MINUIT2 based on MIGRAD id minuit2 exists
   else
      min = new ROOT::Minuit2::Minuit2Minimizer();
#endif

   return min;
}

#endif

ROOT::Math::DistSampler * ROOT::Math::Factory::CreateDistSampler(const std::string & type) {
#ifdef MATH_NO_PLUGIN_MANAGER
   MATH_ERROR_MSG("Factory::CreateDistSampler","ROOT plug-in manager not available");
   return 0;
#else
   // create a DistSampler class using the ROOT plug-in manager
   const char * typeName = type.c_str();
   if (type.empty() )  typeName = ROOT::Math::DistSamplerOptions::DefaultSampler().c_str();

   R__LOCKGUARD2(gROOTMutex);

   TPluginManager *pm = gROOT->GetPluginManager();
   assert(pm != 0);
   TPluginHandler *h = pm->FindHandler("ROOT::Math::DistSampler", typeName );
   if (h != 0) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("Factory::CreateDistSampler","Error loading DistSampler plug-in");
         return 0;
      }

      ROOT::Math::DistSampler * smp = reinterpret_cast<ROOT::Math::DistSampler *>( h->ExecPlugin(0) );
      assert(smp != 0);
      return smp;
   }
   MATH_ERROR_MSGVAL("Factory::CreateDistSampler","Error finding DistSampler plug-in",typeName);
   return 0;
#endif
}


