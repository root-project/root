/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoBoolCombinator.h"
#include "TGeoVolume.h"
#include "TGeoPainter.h"
#include "TGeoCompositeShape.h"

/*************************************************************************
 * TGeoCompositeShape - composite shapes are defined by their list of 
 *   shapes, corresponding transformation matrices and boolean combinator.
 *************************************************************************/

ClassImp(TGeoCompositeShape)

//-----------------------------------------------------------------------------
//TGeoCompositeShape::TGeoCompositeShape()
//{
// Default constructor
//   SetBit(TGeoShape::kGeoComb);
//   fNcomponents = 0;
//   fShapes      = 0;
//   fMatrices    = 0;
//   fCombinator  = 0;
//}   
//-----------------------------------------------------------------------------
TGeoCompositeShape::TGeoCompositeShape()
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetBit(TGeoShape::kGeoComb);
   fNcomponents = 0;
   fShapes      = 0;
   fMatrices    = 0;
   fCombinator  = 0;
}   
//-----------------------------------------------------------------------------
TGeoCompositeShape::~TGeoCompositeShape()
{
// destructor
   if (fShapes) delete fShapes;
   if (fMatrices) delete fMatrices;
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::AddShape(TGeoShape *shape, TGeoMatrix *mat)
{
// add a shape and its transformation matrix to the combination
   if (!fShapes) fShapes = new TList();
   fShapes->Add(shape);
   if (!fMatrices) fMatrices = new TList();
   fMatrices->Add(mat);
   fNcomponents++;
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::SetCombination(TGeoBoolCombinator *comb)
{
// set the boolean combination 
   fCombinator = comb;
}
//-----------------------------------------------------------------------------   
void TGeoCompositeShape::ComputeBBox()
{
// compute bounding box of the sphere
   if(fCombinator) fCombinator->ComputeBBox();
}   
//-----------------------------------------------------------------------------
Bool_t TGeoCompositeShape::Contains(Double_t *point)
{
// test if point is inside this sphere
   if (fCombinator) 
      return fCombinator->Contains(point);
   else
      return kFALSE;
   return kFALSE;
}
//-----------------------------------------------------------------------------
Double_t TGeoCompositeShape::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   if (fCombinator)
      fCombinator->DistToSurf(point, dir);
   else
      return 0.0;
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::InspectShape()
{
// print shape parameters
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::Paint(Option_t *option)
{
// paint this shape according to option
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoCompositeShape::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0;
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::SetPoints(Double_t *buff) const
{
// create points for a composite shape
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::SetPoints(Float_t *buff) const
{
// create points for a composite shape
}

