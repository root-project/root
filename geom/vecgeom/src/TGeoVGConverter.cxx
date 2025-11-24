// @(#)root/vecgeom:$Id:$
// Author: Mihaela Gheata   30/03/16
/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoVGConverter
\ingroup Geometry_classes

Geometry converter to VecGeom
*/

#include "TGeoVGConverter.h"
#include "TGeoVGShape.h"
#include "TGeoBBox.h"
#include "TClass.h"

#include <iostream>

namespace {

Bool_t  isInSelection(const TGeoShape* shape,  const std::set<TGeoShape::EShapeType>& selection) {
   for (auto element : selection) {

      // Special treatment of TGeo::kGeoBox, which is base class for all other shapes
      if (element == TGeoShape::kGeoBox) {
         if (typeid(*shape) == typeid(TGeoBBox) ) {
            return true;
         } else {
            continue;
         }
      }

      // Just test bit for other shapes
      if (shape->TestShapeBit(element)) {
        return true;
      }
   }
   return false;
}

}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TGeoVGConverter::TGeoVGConverter(TGeoManager *manager) : TVirtualGeoConverter(manager)
{
   TVirtualGeoConverter::SetConverter(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor.

TGeoVGConverter::~TGeoVGConverter() {}

////////////////////////////////////////////////////////////////////////////////
/// Main geometry conversion method.
/// Convert all geometry shapes connected to volumes to VecGeom shapes

void TGeoVGConverter::ConvertGeometry()
{
   // First convert the top volume
   TGeoVolume *top = fGeom->GetMasterVolume();
   TGeoVGShape *vgshape = nullptr;
   if (!top->GetShape()->IsVecGeom())
      vgshape = TGeoVGShape::Create(top->GetShape());
   Int_t nconverted = 0;
   // If shape of top volume not known by VecGeom, keep old one
   if (vgshape) {
      nconverted++;
      top->SetShape(vgshape);
   }

   // Uncomment to print info about selected/excluded shapes
   // if ( ! fSelectedShapeTypes.empty()) {
   //   printf("# %zu selected shape type for conversion \n", fSelectedShapeTypes.size());
   // }
   // if ( ! fExcludedShapeTypes.empty()) {
   //   printf("# %zu shape types excluded from conversion \n", fExcludedShapeTypes.size());
   // }

   // Now iterate the active geometry tree
   TGeoIterator next(fGeom->GetTopVolume());
   TGeoNode *node;
   while ((node = next.Next())) {
      TGeoVolume *vol = node->GetVolume();
      // Skip shape if already converted
      if (vol->GetShape()->IsVecGeom())
         continue;

      // If fSelectedShapeTypes is not empty, convert only selected shapes
      if ((! fSelectedShapeTypes.empty()) &&
          (! isInSelection(vol->GetShape(), fSelectedShapeTypes))) {
         printf("# Shape type %s is not selected for conversion\n", vol->GetShape()->IsA()->GetName());
         continue;
      }

      // Skip shapes excluded from conversion
      if (isInSelection(vol->GetShape(), fExcludedShapeTypes)) {
         printf("# Shape type %s is excluded from conversion\n", vol->GetShape()->IsA()->GetName());
         continue;
      }

      // printf("Converting %s\n", vol->GetName());
      vgshape = TGeoVGShape::Create(vol->GetShape());
      if (vgshape) {
         nconverted++;
         vol->SetShape(vgshape);
      }
   }
   printf("# Converted %d shapes to VecGeom ones\n", nconverted);
}

////////////////////////////////////////////////////////////////////////////////
/// Select shape(s) for conversion.
/// Conversion is performed only on selected shapes (if set)

void TGeoVGConverter::SelectShapeType(TGeoShape::EShapeType shapeType)
{
   fSelectedShapeTypes.insert(shapeType);
}

////////////////////////////////////////////////////////////////////////////////
/// Exclude shape(s) from conversion.

void TGeoVGConverter::ExcludeShapeType(TGeoShape::EShapeType shapeType)
{
   fExcludedShapeTypes.insert(shapeType);
}
