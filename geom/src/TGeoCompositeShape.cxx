// @(#)root/geom:$Name:  $:$Id: TGeoCompositeShape.cxx,v 1.6 2002/10/13 15:45:24 brun Exp $
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoBoolNode.h"
#include "TVirtualGeoPainter.h"

#include "TGeoCompositeShape.h"

/*************************************************************************
 * TGeoCompositeShape - composite shapes are defined by their list of 
 *   shapes, corresponding transformation matrices and boolean combinator.
 *************************************************************************/

ClassImp(TGeoCompositeShape)

//-----------------------------------------------------------------------------
TGeoCompositeShape::TGeoCompositeShape()
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetBit(TGeoShape::kGeoComb);
   fNode  = 0;
}   
//-----------------------------------------------------------------------------
TGeoCompositeShape::TGeoCompositeShape(const char *name, const char *expression)
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetBit(TGeoShape::kGeoComb);
   SetName(name);
   fNode  = 0;
   MakeNode(expression);
   if (!fNode) {
      char message[256];
      sprintf(message, "could not build expression %s", expression);
      Error("ctor", message);
      return;
   }
   ComputeBBox();
}  
//-----------------------------------------------------------------------------
TGeoCompositeShape::TGeoCompositeShape(const char *expression)
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetBit(TGeoShape::kGeoComb);
   fNode  = 0;
   MakeNode(expression);
   if (!fNode) {
      char message[256];
      sprintf(message, "could not build expression %s", expression);
      Error("ctor", message);
      return;
   }
   ComputeBBox();
}  
//-----------------------------------------------------------------------------
TGeoCompositeShape::~TGeoCompositeShape()
{
// destructor
   if (fNode) delete fNode;
}
//-----------------------------------------------------------------------------   
void TGeoCompositeShape::ComputeBBox()
{
// compute bounding box of the sphere
   if(fNode) fNode->ComputeBBox(fDX, fDY, fDZ, fOrigin);
}   
//-----------------------------------------------------------------------------
Bool_t TGeoCompositeShape::Contains(Double_t *point) const
{
// test if point is inside this sphere
   if (fNode) return fNode->Contains(point);
   return kFALSE;
}
//-----------------------------------------------------------------------------
Double_t TGeoCompositeShape::DistToIn(Double_t *point, Double_t *dir, Int_t iact,
                                      Double_t step, Double_t *safe) const
{
// Compute distance from outside point to this composite shape.
   if (!CouldBeCrossed(point, dir)) return kBig;
   if (fNode) return fNode->DistToIn(point, dir, iact, step, safe);
   return kBig;
}   
//-----------------------------------------------------------------------------
Double_t TGeoCompositeShape::DistToOut(Double_t *point, Double_t *dir, Int_t iact,
                                      Double_t step, Double_t *safe) const
{
// Compute distance from inside point to outside of this composite shape.
   if (fNode) return fNode->DistToOut(point, dir, iact, step, safe);
   return kBig;
}   
//-----------------------------------------------------------------------------
Double_t TGeoCompositeShape::DistToSurf(Double_t *point, Double_t *dir) const
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoCompositeShape::Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Double_t step) 
{
// Divide all range of iaxis in range/step cells 
   Error("Divide", "Composite shapes cannot be divided");
   return voldiv;
}      
//-----------------------------------------------------------------------------
void TGeoCompositeShape::InspectShape() const
{
// print shape parameters
   printf("*** TGeoCompositeShape : %s = %s\n", GetName(), GetTitle());
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::MakeNode(const char *expression)
{
// Make a booleann node according to the top level boolean operation of expression.
// Propagates signal to branches until expression is fully decomposed.
   printf("Making node for : %s\n", expression);
   if (fNode) delete fNode;
   fNode = 0;
   SetTitle(expression);
   TString sleft, sright, smat;
   Int_t boolop;
   boolop = TGeoManager::Parse(expression, sleft, sright, smat);
   if (boolop<0) {
      // fail
      Error("MakeNode", "parser error");
      return;
   }   
   if (smat.Length())
      Warning("MakeNode", "no geometrical transformation allowed at this level");
   switch (boolop) {
      case 0: 
         Error("MakeNode", "Expression has no boolean operation");
         return;    
      case 1:
         fNode = new TGeoUnion(sleft.Data(), sright.Data());
         return;
      case 2:
         fNode = new TGeoSubtraction(sleft.Data(), sright.Data());
         return;
      case 3:
         fNode = new TGeoIntersection(sleft.Data(), sright.Data());
   }
}               
//-----------------------------------------------------------------------------
void TGeoCompositeShape::NextCrossing(TGeoParamCurve *c, Double_t *point) const
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoHMatrix *glmat = gGeoManager->GetCurrentMatrix();
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   PaintNext(glmat, option);
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::PaintNext(TGeoHMatrix *glmat, Option_t *option)
{
// paint this shape according to option
   if (fNode) fNode->PaintNext(glmat, option);
}
//-----------------------------------------------------------------------------
Double_t TGeoCompositeShape::Safety(Double_t *point, Double_t *spoint, Option_t *option) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::SetPoints(Double_t *buff) const
{
// create points for a composite shape
   TGeoBBox::SetPoints(buff);
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::SetPoints(Float_t *buff) const
{
// create points for a composite shape
   TGeoBBox::SetPoints(buff);
}
//-----------------------------------------------------------------------------
void TGeoCompositeShape::Sizeof3D() const
{
// compute size of this 3D object
   if (fNode) fNode->Sizeof3D();
}
