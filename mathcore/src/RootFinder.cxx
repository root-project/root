// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/RootFinder.h"
#include "Math/BrentRootFinder.h"

#ifndef MATH_NO_PLUGIN_MANAGER

#include "TROOT.h"
#include "TPluginManager.h"

#else // case no plugin manager is available
#ifdef R__HAS_MATHMORE
#include "Math/RootFinderAlgorithms.h"
#endif

#endif

#include <cassert>

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif

namespace ROOT {
namespace Math {


RootFinder::RootFinder(RootFinder::Type type)
{
   fSolver = 0;
   SetMethod(type);
}

int RootFinder::SetMethod(RootFinder::Type type)
{
   if ( type == RootFinder::BRENT )
   {
      fSolver = new BrentRootFinder();
      return 0;
   }

#ifdef MATH_NO_PLUGIN_MANAGER    // no PM available
#ifdef R__HAS_MATHMORE

   switch(type) {

   case GSL_BISECTION:
      fSolver = new ROOT::Math::Roots::Bisection();
      break;
   case GSL_FALSE_POS:
      fSolver = new ROOT::Math::Roots::FalsePos();
      break;
   case GSL_BRENT:
      fSolver = new ROOT::Math::Roots::Brent();
      break;
   case GSL_NEWTON: 
      fSolver = new ROOT::Math::Roots::Newton();
      break;
   case GSL_SECANT: 
      fSolver = new ROOT::Math::Roots::Secant();
      break;
   case GSL_STEFFENSON:
      fSolver = new ROOT::Math::Roots::Steffenson();
      break;
   default:
      MATH_ERROR_MSG("RootFinder::SetMethod","RootFinderMethod type is not available in MathCore");
      fSolver = 0;
      return -1;
      break;
   };

#else
   MATH_ERROR_MSG("RootFinder::SetMethod","RootFinderMethod type is not available in MathCore");
   return -1;
#endif

#else  // case of using Plugin Manager
   TPluginHandler *h;
   std::string stype;

   switch(type) {
      
   case GSL_BISECTION:
      //fSolver = new ROOT::Math::Roots::Bisection();
      stype = "Bisection";
      break;
   case GSL_FALSE_POS:
      //fSolver = new ROOT::Math::Roots::FalsePos();
      stype = "FalsePos";
      break;
   case GSL_BRENT:
      //fSolver = new ROOT::Math::Roots::Brent();
      stype = "Brent";
      break;
   case GSL_NEWTON: 
      //fSolver = new ROOT::Math::Roots::Newton();
      stype = "Newton";
      break;
   case GSL_SECANT: 
      //fSolver = new ROOT::Math::Roots::Secant();
      stype = "Secant";
      break;
   case GSL_STEFFENSON:
      //fSolver = new ROOT::Math::Roots::Steffenson();
      stype = "Steffenson";
      break;
   default:
      MATH_ERROR_MSG("RootFinder::SetMethod","RootFinderMethod type is not available in MathCore");
      fSolver = 0;
      return -1;
      break;
   };

   if ((h = gROOT->GetPluginManager()->FindHandler("ROOT::Math::IRootFinderMethod", stype.c_str() ))) {
      if (h->LoadPlugin() == -1) {
         MATH_ERROR_MSG("RootFinder::SetMethod","Error loading RootFinderMethod");
         return -1;
      }

      fSolver = reinterpret_cast<ROOT::Math::IRootFinderMethod *>( h->ExecPlugin(0) );
      assert(fSolver != 0);
   }
   else {
      MATH_ERROR_MSG("RootFinder::SetMethod","Error loading RootFinderMethod");
      return -1;
   }

#endif

   return 0;
}

// int RootFinder::SetMethod(RootFinder::Type type)
// {
//    if ( fSolver )
//       delete fSolver;

//    switch(type) {

//    case GSL_BISECTION:
//       fSolver = new ROOT::Math::Roots::Bisection();
//       break;
//    case GSL_FALSE_POS:
//       fSolver = new ROOT::Math::Roots::FalsePos();
//       break;
//    case GSL_BRENT:
//       fSolver = new ROOT::Math::Roots::Brent();
//       break;
//    case GSL_NEWTON: 
//       fSolver = new ROOT::Math::Roots::Newton();
//       break;
//    case GSL_SECANT: 
//       fSolver = new ROOT::Math::Roots::Secant();
//       break;
//    case GSL_STEFFENSON:
//       fSolver = new ROOT::Math::Roots::Steffenson();
//       break;

//    case RootFinder::BRENT:
//    default:
//       fSolver = new BrentRootFinder();
//       break;
//    };

//    return 0;
// }

RootFinder::~RootFinder()
{
   delete fSolver;
}


}
}
