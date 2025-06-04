/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualGeoChecker
\ingroup Geometry_classes

Abstract class for geometry checkers
*/

#include "TROOT.h"
#include "TVirtualGeoChecker.h"
#include "TPluginManager.h"
#include "TGeoManager.h"

TVirtualGeoChecker *TVirtualGeoChecker::fgGeoChecker = nullptr;

////////////////////////////////////////////////////////////////////////////////
/// Geometry checker default constructor

TVirtualGeoChecker::TVirtualGeoChecker() {}

////////////////////////////////////////////////////////////////////////////////
/// Geometry checker destructor

TVirtualGeoChecker::~TVirtualGeoChecker()
{
   fgGeoChecker = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function returning a pointer to the geometry checker.
/// If the geometry checker does not exist a default checker is created.

TVirtualGeoChecker *TVirtualGeoChecker::GeoChecker()
{
   // if no painter set yet, create a default painter via the PluginManager
   if (!fgGeoChecker) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGeoChecker"))) {
         if (h->LoadPlugin() == -1)
            return nullptr;
         fgGeoChecker = (TVirtualGeoChecker *)h->ExecPlugin(1, gGeoManager);
      }
   }
   return fgGeoChecker;
}
