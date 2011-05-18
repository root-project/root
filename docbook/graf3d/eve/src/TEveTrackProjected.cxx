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
#include "TEveTrans.h"

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
   TEveTrack (),
   fOrigPnts (0)
{
   // Default constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   // This is virtual method from base-class TEveProjected.

   TEveProjected::SetProjection(mng, model);
   CopyVizParams(dynamic_cast<TEveElement*>(model));

   TEveTrack* otrack = dynamic_cast<TEveTrack*>(fProjectable);
   SetTrackParams(*otrack);
   SetLockPoints(otrack->GetLockPoints());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::SetDepthLocal(Float_t d)
{
   // Set depth (z-coordinate) of the projected points.

   SetDepthCommon(d, this, fBBox);

   Int_t    n = Size();
   Float_t *p = GetP() + 2;
   for (Int_t i = 0; i < n; ++i, p+=3)
   {
      *p = fDepth;
   }

   for (vPathMark_i pm = fPathMarks.begin(); pm != fPathMarks.end(); ++pm)
   {
      pm->fV.fZ = fDepth;
   }
}

//______________________________________________________________________________
void TEveTrackProjected::UpdateProjection()
{
   // Virtual method from base-class TEveProjected.

   MakeTrack(kFALSE); // TEveProjectionManager makes recursive calls
}

//______________________________________________________________________________
void TEveTrackProjected::GetBreakPoint(Int_t idx, Bool_t back,
                                       Float_t& x, Float_t& y, Float_t& z)
{
   // With bisection calculate break-point vertex.

   TEveProjection *projection = fManager->GetProjection();

   TEveVector vL = fOrigPnts[idx];
   TEveVector vR = fOrigPnts[idx+1];
   TEveVector vM, vLP, vMP;
   while ((vL-vR).Mag2() > 1e-10f)
   {
      vM.Mult(vL+vR, 0.5f);
      vLP.Set(vL); projection->ProjectPoint(vLP.fX, vLP.fY, vLP.fZ, 0);
      vMP.Set(vM); projection->ProjectPoint(vMP.fX, vMP.fY, vMP.fZ, 0);
      if (projection->AcceptSegment(vLP, vMP, 0.0f))
      {
         vL.Set(vM);
      }
      else
      {
         vR.Set(vM);
      }
   }

   if (back) {
      x = vL.fX; y = vL.fY; z = vL.fZ;
   } else {
      x = vR.fX; y = vR.fY; z = vR.fZ;
   }
   projection->ProjectPoint(x, y, z, fDepth);
}

//______________________________________________________________________________
Int_t TEveTrackProjected::GetBreakPointIdx(Int_t start)
{
   // Findex index of the last point that lies within the same
   // segment of projected space.
   // For example, rho-z projection separates upper and lower hemisphere
   // and tracks break into two lines when crossing the y=0 plane.

   TEveProjection *projection = fManager->GetProjection();

   Int_t val = fLastPoint;

   TEveVector v1, v2;
   if (Size() > 1)
   {
      Int_t i = start;
      while(i < fLastPoint)
      {
         GetPoint(i,   v1.fX, v1.fY, v1.fZ);
         GetPoint(i+1, v2.fX, v2.fY, v2.fZ);
         if(projection->AcceptSegment(v1, v2, fPropagator->GetDelta()) == kFALSE)
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

   TEveTrack      *otrack     = dynamic_cast<TEveTrack*>(fProjectable);
   TEveTrans      *trans      = otrack->PtrMainTrans(kFALSE);
   TEveProjection *projection = fManager->GetProjection();

   fBreakPoints.clear();

   fPathMarks.clear();
   SetPathMarks(*otrack);
   if (GetLockPoints() || otrack->Size() > 0)
   {
      ClonePoints(*otrack);
      fLastPMIdx = otrack->GetLastPMIdx();
   }
   else
   {
      TEveTrack::MakeTrack(recurse);
   }
   if (Size() == 0) return; // All points can be outside of MaxR / MaxZ limits.

   // Break segments additionally if required by the projection.
   ReduceSegmentLengths(projection->GetMaxTrackStep());

   // Project points, store originals (needed for break-points).
   Float_t *p = GetP();
   fOrigPnts  = new TEveVector[Size()];
   for (Int_t i = 0; i < Size(); ++i, p+=3)
   {
      if (trans) trans->MultiplyIP(p);
      fOrigPnts[i].Set(p);
      projection->ProjectPointfv(p, fDepth);
   }

   Float_t x, y, z;
   Int_t   bL = 0, bR = GetBreakPointIdx(0);
   std::vector<TEveVector> vvec;
   while (kTRUE)
   {
      for (Int_t i=bL; i<=bR; i++)
      {
         GetPoint(i, x, y, z);
         vvec.push_back(TEveVector(x, y, z));
      }
      if (bR == fLastPoint)
         break;

      GetBreakPoint(bR, kTRUE,  x, y, z); vvec.push_back(TEveVector(x, y, z));
      fBreakPoints.push_back((Int_t)vvec.size());
      GetBreakPoint(bR, kFALSE, x, y, z); vvec.push_back(TEveVector(x, y, z));

      bL = bR + 1;
      bR = GetBreakPointIdx(bL);
   }
   fBreakPoints.push_back((Int_t)vvec.size()); // Mark the track-end for drawing.

   // Decide if points need to be fixed.
   // This (and the fixing itself) should really be done in TEveProjection but
   // for now we do it here as RhoZ is the only one that needs it.
   Bool_t  fix_y  = kFALSE;
   Float_t sign_y = 0;
   if (projection->HasSeveralSubSpaces())
   {
      switch (fPropagator->GetProjTrackBreaking())
      {
         case TEveTrackPropagator::kPTB_UseFirstPointPos:
         {
            fix_y  = kTRUE;
            sign_y = vvec.front().fY;
            break;
         }
         case TEveTrackPropagator::kPTB_UseLastPointPos:
         {
            fix_y  = kTRUE;
            sign_y = vvec.back().fY;
            break;
         }
      }
   }

   Reset((Int_t)vvec.size());
   for (std::vector<TEveVector>::iterator i=vvec.begin(); i!=vvec.end(); ++i)
   {
      if (fix_y)
         SetNextPoint((*i).fX, TMath::Sign((*i).fY, sign_y), (*i).fZ);
      else
         SetNextPoint((*i).fX, (*i).fY, (*i).fZ);
   }
   delete [] fOrigPnts;

   // Project path-marks
   for (vPathMark_i pm = fPathMarks.begin(); pm != fPathMarks.end(); ++pm)
   {
      projection->ProjectVector(trans, pm->fV, fDepth);
   }
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
   CopyVizParams(dynamic_cast<TEveElement*>(model));

   TEveTrackList& tl = * dynamic_cast<TEveTrackList*>(model);
   SetPropagator(tl.GetPropagator());
}

//______________________________________________________________________________
void TEveTrackListProjected::SetDepthLocal(Float_t /*d*/)
{
   // This is not needed for functionality as SetDepth(Float_t d)
   // is overriden -- but SetDepthLocal() is abstract.
   // Just emits a warning if called.

   Warning("SetDepthLocal", "This function only exists to fulfill an abstract interface.");
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
   for (List_i i = el->BeginChildren(); i != el->EndChildren(); ++i)
   {
      ptrack = dynamic_cast<TEveTrackProjected*>(*i);
      if (ptrack)
         ptrack->SetDepth(d);
      if (fRecurse)
         SetDepth(d, *i);
   }
}
