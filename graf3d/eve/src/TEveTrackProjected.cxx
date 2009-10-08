// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTrackProjected.h"
#include "TEveTrackPropagator.h"
#include "TEveProjectionManager.h"
#include "TEveVSDStructs.h"

//==============================================================================
//==============================================================================
// TEveTrackProjected
//==============================================================================

//______________________________________________________________________________
//
// Projected copy of a TEveTrack.

ClassImp(TEveTrackProjected);

//______________________________________________________________________________
TEveTrackProjected::TEveTrackProjected() :
   TEveTrack  (),
   fOrigPnts  (0),
   fProjection(0)
{
   // Default constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   // This is virtual method from base-class TEveProjected.

   TEveProjected::SetProjection(mng, model);
   TEveTrack* origTrack = dynamic_cast<TEveTrack*>(fProjectable);

   SetTrackParams(*origTrack);
   SetPathMarks  (*origTrack);

   SetLockPoints(origTrack->GetLockPoints());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::SetDepth(Float_t d)
{
   // Set depth (z-coordinate) of the projected points.

   SetDepthCommon(d, this, fBBox);

   Int_t    n = Size();
   Float_t *p = GetP();
   for (Int_t i = 0; i < n; ++i, p+=3)
      p[2] = fDepth;

   // !!!! Missing path-marks move. But they are not projected anyway
}

//______________________________________________________________________________
void TEveTrackProjected::UpdateProjection()
{
   // Virtual method from base-class TEveProjected.

   fProjection = fManager->GetProjection();
   MakeTrack(kFALSE); // TEveProjectionManager makes recursive calls
}

//______________________________________________________________________________
void TEveTrackProjected::GetBreakPoint(Int_t idx, Bool_t back,
                                       Float_t& x, Float_t& y, Float_t& z)
{
   // With bisection calculate break-point vertex.

   TEveVector vL = fOrigPnts[idx];
   TEveVector vR = fOrigPnts[idx+1];
   TEveVector vM, vLP, vMP;
   while ((vL-vR).Mag() > 0.01)
   {
      vM.Mult(vL+vR, 0.5f);
      vLP.Set(vL); fProjection->ProjectPoint(vLP.fX, vLP.fY, vLP.fZ);
      vMP.Set(vM); fProjection->ProjectPoint(vMP.fX, vMP.fY, vMP.fZ);
      if (fProjection->AcceptSegment(vLP, vMP, 0.0f))
      {
         vL.Set(vM);
      }
      else
      {
         vR.Set(vM);
      }
   }

   if(back) {
      x = vL.fX; y = vL.fY; z = vL.fZ;
   } else {
      x = vR.fX; y = vR.fY; z = vR.fZ;
   }
   fProjection->ProjectPoint(x, y, z);
}

//______________________________________________________________________________
Int_t TEveTrackProjected::GetBreakPointIdx(Int_t start)
{
   // Findex index of the last point that lies within the same
   // segment of projected space.
   // For example, rho-z projection separates upper and lower hemisphere
   // and tracks break into two lines when crossing the y=0 plane.

   Int_t val = fLastPoint;

   TEveVector v1, v2;
   if (Size() > 1)
   {
      Int_t i = start;
      while(i < fLastPoint)
      {
         GetPoint(i,   v1.fX, v1.fY, v1.fZ);
         GetPoint(i+1, v2.fX, v2.fY, v2.fZ);
         if(fProjection->AcceptSegment(v1, v2, fPropagator->GetDelta()) == kFALSE)
         {
            val = i;
            break;
         }
         i++;
      }
   }
   return val;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::MakeTrack(Bool_t recurse)
{
   // Calculate the points of the track for drawing.
   // Call base-class, project, find break-points and insert points
   // required for full representation.

   fBreakPoints.clear();

   if (GetLockPoints())
   {
      ClonePoints(*dynamic_cast<TEveTrack*>(fProjectable));
   }
   else
   {
      TEveTrack::MakeTrack(recurse);
   }
   if (Size() == 0) return; // All points can be outside of MaxR / MaxZ limits.

   // Break segments additionally if required by the projection.
   ReduceSegmentLengths(fProjection->GetMaxTrackStep());

   // Project points, store originals (needed for break-points).
   Float_t *p = GetP();
   fOrigPnts  = new TEveVector[Size()];
   for (Int_t i = 0; i < Size(); ++i, p+=3)
   {
      fOrigPnts[i].Set(p);
      fProjection->ProjectPoint(p[0], p[1], p[2]);
      p[2] = fDepth;
   }

   Float_t x, y, z;
   Int_t   bL = 0, bR = GetBreakPointIdx(0);
   Int_t   sign = 1;
   Bool_t  break_track = ShouldBreakTrack();
   std::vector<TEveVector> vvec;
   while (kTRUE)
   {
      for(Int_t i=bL; i<=bR; i++)
      {
         GetPoint(i, x, y, z);
         vvec.push_back(TEveVector(x, sign*y, z));
      }
      if (bR == fLastPoint)
         break;

      if (break_track)
      {
         GetBreakPoint(bR, kTRUE,  x, y, z); vvec.push_back(TEveVector(x, y, z));
         fBreakPoints.push_back((Int_t)vvec.size());
         GetBreakPoint(bR, kFALSE, x, y, z); vvec.push_back(TEveVector(x, y, z));
      }
      else
      {
         sign = -sign;
      }
      bL = bR + 1;
      bR = GetBreakPointIdx(bL);
   }
   fBreakPoints.push_back((Int_t)vvec.size()); // Mark the track-end for drawing.

   Reset((Int_t)vvec.size());
   for (std::vector<TEveVector>::iterator i=vvec.begin(); i!=vvec.end(); ++i)
      SetNextPoint((*i).fX, (*i).fY, (*i).fZ);
   delete [] fOrigPnts;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::PrintLineSegments()
{
   // Print line segments info.

   printf("%s LineSegments:\n", GetName());
   Int_t start = 0;
   Int_t segment = 0;
   TEveVector sVec;
   TEveVector bPnt;
   for (std::vector<Int_t>::iterator bpi = fBreakPoints.begin();
        bpi != fBreakPoints.end(); ++bpi)
   {
      Int_t size = *bpi - start;

      GetPoint(start, sVec.fX, sVec.fY, sVec.fZ);
      GetPoint((*bpi)-1, bPnt.fX, bPnt.fY, bPnt.fZ);
      printf("seg %d size %d start %d ::(%f, %f, %f) (%f, %f, %f)\n",
             segment, size, start, sVec.fX, sVec.fY, sVec.fZ,
             bPnt.fX, bPnt.fY, bPnt.fZ);
      start   += size;
      segment ++;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::SecSelected(TEveTrack* /*track*/)
{
    // Virtual method from from base-class TEveTrack.

   TEveTrack* t = dynamic_cast<TEveTrack*>(fProjectable);
   if (t)
      t->SecSelected(t);
}


//==============================================================================
//==============================================================================
// TEveTrackListProjected
//==============================================================================

//______________________________________________________________________________
//
// Specialization of TEveTrackList for holding TEveTrackProjected objects.

ClassImp(TEveTrackListProjected);

//______________________________________________________________________________
TEveTrackListProjected::TEveTrackListProjected() :
   TEveTrackList (),
   TEveProjected ()
{
   // Default constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackListProjected::SetProjection(TEveProjectionManager* proj, TEveProjectable* model)
{
   // This is virtual method from base-class TEveProjected.

   TEveProjected::SetProjection(proj, model);

   TEveTrackList& tl = * dynamic_cast<TEveTrackList*>(model);
   SetLineColor(tl.GetLineColor());
   SetLineStyle(tl.GetLineStyle());
   SetLineWidth(tl.GetLineWidth());
   SetMarkerColor(tl.GetMarkerColor());
   SetMarkerStyle(tl.GetMarkerStyle());
   SetMarkerSize (tl.GetMarkerSize());
   SetRnrLine  (tl.GetRnrLine());
   SetRnrPoints(tl.GetRnrPoints());

   SetPropagator(tl.GetPropagator());
}

//______________________________________________________________________________
void TEveTrackListProjected::SetDepth(Float_t d)
{
   // Set depth of all children inheriting from TEveTrackProjected.

   SetDepth(d, this);
}

//______________________________________________________________________________
void TEveTrackListProjected::SetDepth(Float_t d, TEveElement* el)
{
   // Set depth of all children of el inheriting from TEveTrackProjected.

   TEveTrackProjected* ptrack;
   for (List_i i=el->BeginChildren(); i!=el->EndChildren(); ++i)
   {
      ptrack = dynamic_cast<TEveTrackProjected*>(*i);
      if (ptrack)
         ptrack->SetDepth(d);
      if (fRecurse)
         SetDepth(d, *i);
   }
}
