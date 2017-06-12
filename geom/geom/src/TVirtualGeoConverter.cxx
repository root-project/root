// @(#)root/geom:$Id$
// Author: Mihaela Gheata   30/03/16

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualGeoConverter
\ingroup Geometry_classes

Abstract class for geometry converters
*/

#include "TVirtualGeoConverter.h"

#include "TError.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TGeoManager.h"

TVirtualGeoConverter  *TVirtualGeoConverter::fgGeoConverter = 0;

ClassImp(TVirtualGeoConverter);

////////////////////////////////////////////////////////////////////////////////
/// Geometry converter default constructor

TVirtualGeoConverter::TVirtualGeoConverter(TGeoManager *geom)
    :TObject(), fGeom(geom)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Geometry converter default destructor

TVirtualGeoConverter::~TVirtualGeoConverter()
{
   fgGeoConverter = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Static function returning a pointer to the current geometry converter.
/// The converter implements the ConvertGeometry function.
/// If the geometry converter does not exist a default converter is created.

TVirtualGeoConverter *TVirtualGeoConverter::Instance(TGeoManager *geom)
{
   // if no converter set yet, create a default converter via the PluginManager
   TGeoManager *mgr = geom;
   if (!mgr) mgr = gGeoManager;
   if (!fgGeoConverter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGeoConverter"))) {
         if (h->LoadPlugin() == -1) {
            ::Error("TVirtualGeoConverter::Instance()",
            "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
            "It appears that you are missing or having outdated support for VecGeom package. "
            "To enable it, configure ROOT with:\n"
            "   -Dvecgeom -DCMAKE_PREFIX_PATH=<vecgeom_prefix_path>/lib/CMake/VecGeom"
            "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
            return 0;
         }
         fgGeoConverter = (TVirtualGeoConverter*)h->ExecPlugin(1,mgr);
      }
   }
   if (fgGeoConverter) fgGeoConverter->SetGeometry(mgr);
   return fgGeoConverter;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function to set an alternative converter.

void TVirtualGeoConverter::SetConverter(const TVirtualGeoConverter *converter)
{
   fgGeoConverter = (TVirtualGeoConverter*)converter;
}
