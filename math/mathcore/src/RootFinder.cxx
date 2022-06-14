// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/RootFinder.h"
#include "Math/IRootFinderMethod.h"
#include "Math/BrentRootFinder.h"

#include "RConfigure.h"

#ifndef MATH_NO_PLUGIN_MANAGER

#include "TROOT.h"
#include "TPluginManager.h"

#else // case no plugin manager is available
#ifdef R__HAS_MATHMORE
#include "Math/RootFinderAlgorithms.h"
#endif

#endif

#include <cassert>

#include "Math/Error.h"

namespace ROOT {
namespace Math {


RootFinder::RootFinder(RootFinder::EType type) :
   fSolver(0)
{
   // constructor passing type (default is kBRENT)
   SetMethod(type);
}

bool RootFinder::SetMethod(RootFinder::EType type)
{
   // set method - Use the plug-in manager if method is implemented in MathMore
   if ( type == RootFinder::kBRENT )
   {
      fSolver = new BrentRootFinder();
      return true;
   }

#ifdef MATH_NO_PLUGIN_MANAGER    // no PM available
#ifdef R__HAS_MATHMORE

   switch(type) {

   case kGSL_BISECTION:
      fSolver = new ROOT::Math::Roots::Bisection();
      break;
   case kGSL_FALSE_POS:
      fSolver = new ROOT::Math::Roots::FalsePos();
      break;
   case kGSL_BRENT:
      fSolver = new ROOT::Math::Roots::Brent();
      break;
   case kGSL_NEWTON:
      fSolver = new ROOT::Math::Roots::Newton();
      break;
   case kGSL_SECANT:
      fSolver = new ROOT::Math::Roots::Secant();
      break;
   case kGSL_STEFFENSON:
      fSolver = new ROOT::Math::Roots::Steffenson();
      break;
   default:
      MATH_ERROR_MSG("RootFinder::SetMethod","RootFinderMethod type is not available in MathCore");
      fSolver = 0;
      return false;
      break;
   };

#else
   MATH_ERROR_MSG("RootFinder::SetMethod","RootFinderMethod type is not available in MathCore");
   return false;
#endif

#else  // case of using Plugin Manager
   TPluginHandler *h;
   std::string stype;

   switch(type) {

   case kGSL_BISECTION:
      stype = "Bisection";
      break;
   case kGSL_FALSE_POS:
      stype = "FalsePos";
      break;
   case kGSL_BRENT:
      stype = "Brent";
      break;
   case kGSL_NEWTON:
      stype = "Newton";
      break;
   case kGSL_SECANT:
      stype = "Secant";
      break;
   case kGSL_STEFFENSON:
      stype = "Steffenson";
      break;
   default:
      MATH_ERROR_MSG("RootFinder::SetMethod","RootFinderMethod type is not available in MathCore");
      fSolver = 0;
      return false;
      break;
   };

   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::IRootFinderMethod", stype.c_str() ))) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("RootFinder::SetMethod","Error loading RootFinderMethod");
         return false;
      }

      fSolver = reinterpret_cast<ROOT::Math::IRootFinderMethod *>( h->ExecPlugin(0) );
      assert(fSolver != 0);
   }
   else {
      MATH_ERROR_MSG("RootFinder::SetMethod","Error loading RootFinderMethod");
      return false;
   }

#endif

   return true;
}


RootFinder::~RootFinder()
{
   // destructor
   delete fSolver;
}


}
}
