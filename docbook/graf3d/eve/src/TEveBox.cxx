// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2010

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveBox.h"
#include "TEveProjectionManager.h"

//==============================================================================
// TEveBox
//==============================================================================

//______________________________________________________________________________
//
// 3D box with arbitrary vertices (cuboid).
// Vertices 0-3 specify the "bottom" rectangle in clockwise direction and
// vertices 4-7 the "top" rectangle so that 4 is above 0, 5 above 1 and so on.
//
// If vertices are provided some local coordinates the transformation matrix
// of the element should also be set (but then the memory usage is increased
// by the size of the TEveTrans object).
//
// Currently only supports 3D -> 2D projections.

ClassImp(TEveBox);

//______________________________________________________________________________
TEveBox::TEveBox(const char* n, const char* t) :
   TEveShape(n, t)
{
   // Constructor.
}

//______________________________________________________________________________
TEveBox::~TEveBox()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveBox::SetVertex(Int_t i, Float_t x, Float_t y, Float_t z)
{
   // Set vertex 'i'.

   fVertices[i][0] = x;
   fVertices[i][1] = y;
   fVertices[i][2] = z;
   ResetBBox();
}

//______________________________________________________________________________
void TEveBox::SetVertex(Int_t i, const Float_t* v)
{
   // Set vertex 'i'.

   fVertices[i][0] = v[0];
   fVertices[i][1] = v[1];
   fVertices[i][2] = v[2];
   ResetBBox();
}

//______________________________________________________________________________
void TEveBox::SetVertices(const Float_t* vs)
{
   // Set vertices.

   memcpy(fVertices, vs, sizeof(fVertices));
   ResetBBox();
}

//==============================================================================

//______________________________________________________________________________
void TEveBox::ComputeBBox()
{
   // Compute bounding-box of the data.

   TEveShape::CheckAndFixBoxOrientationFv(fVertices);

   BBoxInit();
   for (Int_t i=0; i<8; ++i)
   {
      BBoxCheckPoint(fVertices[i]);
   }
}

//______________________________________________________________________________
TClass* TEveBox::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, return TEveBoxProjected class.

   return TEveBoxProjected::Class();
}


//==============================================================================
// TEveBoxProjected
//==============================================================================

//______________________________________________________________________________
//
// Projection of TEveBox.

ClassImp(TEveBoxProjected);

Bool_t TEveBoxProjected::fgDebugCornerPoints = kFALSE;

//______________________________________________________________________________
TEveBoxProjected::TEveBoxProjected(const char* n, const char* t) :
   TEveShape(n, t),
   fBreakIdx(0)
{
   // Constructor.
}

//______________________________________________________________________________
TEveBoxProjected::~TEveBoxProjected()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveBoxProjected::ComputeBBox()
{
   // Compute bounding-box, virtual from TAttBBox.

   BBoxInit();
   for (vVector2_i i = fPoints.begin(); i != fPoints.end(); ++i)
   {
      BBoxCheckPoint(i->fX, i->fY, fDepth);
   }
}

//______________________________________________________________________________
void TEveBoxProjected::SetDepthLocal(Float_t d)
{
   // This is virtual method from base-class TEveProjected.

   SetDepthCommon(d, this, fBBox);
}

//______________________________________________________________________________
void TEveBoxProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   // This is virtual method from base-class TEveProjected.

   TEveProjected::SetProjection(mng, model);
   CopyVizParams(dynamic_cast<TEveElement*>(model));
}

//______________________________________________________________________________
void TEveBoxProjected::UpdateProjection()
{
   // Re-project the box. Projects all points and finds 2D convex-hull.
   //
   // The only issue is with making sure that initial conditions for
   // hull-search are reasonable -- that is, there are no overlaps with the
   // first point.

   TEveBox *box = dynamic_cast<TEveBox*>(fProjectable);

   fDebugPoints.clear();

   // Project points in global CS, remove overlaps.
   vVector2_t pp[2];
   {
      TEveProjection *projection = fManager->GetProjection();
      TEveTrans      *trans      = box->PtrMainTrans(kFALSE);

      TEveVector pbuf;
      for (Int_t i = 0; i < 8; ++i)
      {
         projection->ProjectPointfv(trans, box->GetVertex(i), pbuf, fDepth);
         vVector2_t& ppv = pp[projection->SubSpaceId(pbuf)];

         TEveVector2 p(pbuf);
         Bool_t      overlap = kFALSE;
         for (vVector2_i j = ppv.begin(); j != ppv.end(); ++j)
         {
            if (p.SquareDistance(*j) < TEveProjection::fgEpsSqr)
            {
               overlap = kTRUE;
               break;
            }
         }
         if (! overlap)
         {
            ppv.push_back(p);
            if (fgDebugCornerPoints)
               fDebugPoints.push_back(p);
         }
      }
   }

   fPoints.clear();
   fBreakIdx = 0;

   if ( ! pp[0].empty())
   {
      FindConvexHull(pp[0], fPoints, this);
   }
   if ( ! pp[1].empty())
   {
      fBreakIdx = fPoints.size();
      FindConvexHull(pp[1], fPoints, this);
   }
}

//______________________________________________________________________________
Bool_t TEveBoxProjected::GetDebugCornerPoints()
{
   // Get state of fgDebugCornerPoints static.

   return fgDebugCornerPoints;
}

//______________________________________________________________________________
void TEveBoxProjected::SetDebugCornerPoints(Bool_t d)
{
   // Set state of fgDebugCornerPoints static.
   // When this is true, points will be drawn at the corners of
   // computed convex hull.

   fgDebugCornerPoints = d;
}
