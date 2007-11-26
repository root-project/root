// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveTrackProjected.h>
#include <TEveTrackPropagator.h>
#include <TEveProjectionManager.h>
#include <TEveVSDStructs.h>

//______________________________________________________________________________
// TEveTrackProjected
//
// Projected copy of a TEveTrack.

ClassImp(TEveTrackProjected)

//______________________________________________________________________________
TEveTrackProjected::TEveTrackProjected() :
   TEveTrack     (),
   fOrigPnts(0),
   fProjection(0)
{
   // Default constructor.
}

//______________________________________________________________________________
TEveTrackProjected::~TEveTrackProjected()
{
   // Destructor. Noop.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::SetProjection(TEveProjectionManager* proj, TEveProjectable* model)
{
   TEveProjected::SetProjection(proj, model);
   TEveTrack* origTrack = dynamic_cast<TEveTrack*>(fProjectable);

   SetTrackParams(*origTrack);
   SetPathMarks  (*origTrack);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::UpdateProjection()
{
   fProjection = fProjector->GetProjection();
   MakeTrack(kFALSE); // TEveProjectionManager makes recursive calls
}

//______________________________________________________________________________
void TEveTrackProjected::GetBreakPoint(Int_t idx, Bool_t back,
                                       Float_t& x, Float_t& y, Float_t& z)
{
   TEveVector vL = fOrigPnts[idx];
   TEveVector vR = fOrigPnts[idx+1];
   TEveVector vM, vLP, vMP;
   while((vL-vR).Mag() > 0.01)
   {
      vM.Mult(vL+vR, 0.5f);
      vLP.Set(vL); fProjection->ProjectPoint(vLP.x, vLP.y, vLP.z);
      vMP.Set(vM); fProjection->ProjectPoint(vMP.x, vMP.y, vMP.z);
      if(fProjection->AcceptSegment(vLP, vMP, 0.0f))
      {
         vL.Set(vM);
      }
      else
      {
         vR.Set(vM);
      }
      //printf("new interval Mag %f (%f, %f, %f)(%f, %f, %f) \n",(vL-vR).Mag(), vL.x, vL.y, vL.z, vR.x, vR.y, vR.z);
   }

   if(back)
   {
      x = vL.x; y = vL.y; z = vL.z;
   }
   else
   {
      x = vR.x; y = vR.y; z = vR.z;
   }
   fProjection->ProjectPoint(x, y, z);
   // printf("TEveTrackProjected::GetBreakPoint %d (%f, %f, %f) \n", idx, x, y, z);
}

//______________________________________________________________________________
Int_t  TEveTrackProjected::GetBreakPointIdx(Int_t start)
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
         GetPoint(i,   v1.x, v1.y, v1.z);
         GetPoint(i+1, v2.x, v2.y, v2.z);
         if(fProjection->AcceptSegment(v1, v2, fPropagator->fDelta) == kFALSE)
         {
            val = i;
            break;
         }
         i++;
      }
   }
   // printf("BreakPoint IDX start:%d, BREAK %d,  total:%d \n", start, val, Size());
   return val;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::MakeTrack(Bool_t recurse)
{
   // Calculate the points of the track for drawing.
   // Call base-class, project, find break-points and insert points
   // required for full representation.

   TEveTrack::MakeTrack(recurse);

   fBreakPoints.clear();
   if(Size() == 0) return; // All points can be outside of MaxR / MaxZ limits.

   // Project points, store originals (needed for break-points).
   Float_t *p = GetP();
   fOrigPnts  = new TEveVector[Size()];
   for(Int_t i = 0; i < Size(); ++i, p+=3)
   {
      fOrigPnts[i].Set(p);
      fProjection->ProjectPoint(p[0], p[1], p[2]);
      p[2] = fDepth;
   }

   Float_t x, y, z;
   std::vector<TEveVector> vvec;
   Int_t bL = 0, bR = GetBreakPointIdx(0);
   while (1)
   {
      for(Int_t i=bL; i<=bR; i++)
      {
         GetPoint(i, x, y, z);
         vvec.push_back(TEveVector(x, y, z));
      }
      if (bR == fLastPoint)
         break;

      GetBreakPoint(bR, kTRUE,  x, y, z); vvec.push_back(TEveVector(x, y, z));
      fBreakPoints.push_back(vvec.size());
      GetBreakPoint(bR, kFALSE, x, y, z); vvec.push_back(TEveVector(x, y, z));

      bL = bR + 1;
      bR = GetBreakPointIdx(bL);
   }
   fBreakPoints.push_back(vvec.size()); // Mark the track-end for drawing.

   Reset(vvec.size());
   for (std::vector<TEveVector>::iterator i=vvec.begin(); i!=vvec.end(); ++i)
      SetNextPoint((*i).x, (*i).y, (*i).z);
   delete [] fOrigPnts;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::PrintLineSegments()
{
   printf("%s LineSegments:\n", GetName());
   Int_t start = 0;
   Int_t segment = 0;
   TEveVector S;
   TEveVector E;
   for (std::vector<Int_t>::iterator bpi = fBreakPoints.begin();
        bpi != fBreakPoints.end(); ++bpi)
   {
      Int_t size = *bpi - start;

      GetPoint(start, S.x, S.y, S.z);
      GetPoint((*bpi)-1, E.x, E.y, E.z);
      printf("seg %d size %d start %d ::(%f, %f, %f) (%f, %f, %f)\n",
             segment, size, start, S.x, S.y, S.z, E.x, E.y, E.z);
      start   += size;
      segment ++;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjected::CtrlClicked(TEveTrack* /*track*/)
{
   TEveTrack* t = dynamic_cast<TEveTrack*>(fProjectable);
   if (t)
      t->CtrlClicked(t);
}


//______________________________________________________________________________
// TEveTrackListProjected
//
// Specialization of TEveTrackList for holding TEveTrackProjected objects.

//______________________________________________________________________________
ClassImp(TEveTrackListProjected)

//______________________________________________________________________________
TEveTrackListProjected::TEveTrackListProjected() :
   TEveTrackList    (),
   TEveProjected ()
{
   // Default constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackListProjected::SetProjection(TEveProjectionManager* proj, TEveProjectable* model)
{
   TEveProjected::SetProjection(proj, model);

   TEveTrackList& tl   = * dynamic_cast<TEveTrackList*>(model);
   SetLineColor(tl.GetLineColor());
   SetLineStyle(tl.GetLineStyle());
   SetLineWidth(tl.GetLineWidth());
   SetMarkerColor(tl.GetMarkerColor());
   SetMarkerStyle(tl.GetMarkerStyle());
   SetMarkerSize(tl.GetMarkerSize());
   SetRnrLine(tl.GetRnrLine());
   SetRnrPoints(tl.GetRnrPoints());

   SetPropagator(tl.GetPropagator());
}
