// @(#)root/geomconverter:$Id:$
// Author: Mihaela Gheata   30/03/16
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
// TGeoVGConverter - Geometry converter to VecGeom
//______________________________________________________________________________

#include "TGeoVGConverter.h"
#include "TGeoVGShape.h"

ClassImp(TGeoVGConverter)

////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Geometry converter default constructor*-*-*-*-*-*-*-*-*
///*-*                  ====================================

TGeoVGConverter::TGeoVGConverter(TGeoManager *manager) : TVirtualGeoConverter(manager)
{
   TVirtualGeoConverter::SetConverter(this);
   if (manager) fGeoManager = manager;
   else {
      Error("ctor", "No geometry loaded");
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Geometry converter default destructor*-*-*-*-*-*-*-*-*
///*-*                  ===================================

TGeoVGConverter::~TGeoVGConverter()
{
}

////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Main geometry convertion method *-*-*-*-*-*-*-*-*
///*-*                  ===================================
void TGeoVGConverter::ConvertGeometry()
{
   // Convert all geometry shapes connected to volumes to VecGeom shapes
   TGeoVolume *top = fGeoManager->GetTopVolume();
   TGeoVGShape *placed = new TGeoVGShape(top->GetShape());
   top->SetShape(placed);
   TGeoIterator next(fGeoManager->GetTopVolume());
   TGeoNode *node;
   while ((node = next.Next())) {
      TGeoVolume *vol = node->GetVolume();
      if ( !dynamic_cast<TGeoVGShape*>(vol->GetShape()) ) {
         printf("Converting %s\n", vol->GetName());
         placed = new TGeoVGShape(vol->GetShape());
         if (!placed->GetVGShape()) {
            delete placed;
            gGeoManager->GetListOfShapes()->RemoveLast();
         }   
         else vol->SetShape(placed);
      }
   }
}
