// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTrackGL.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"

/** \class TEveTrackGL
\ingroup TEve
GL-renderer for TEveTrack class.
*/

ClassImp(TEveTrackGL);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TEveTrackGL::TEveTrackGL() : TEveLineGL()
{
   // fDLCache = false; // Disable display list.
}

////////////////////////////////////////////////////////////////////////////////
/// Set model object.

Bool_t TEveTrackGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   TEveLineGL::SetModel(obj);
   fTrack = DynCast<TEveTrack>(obj);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Processes secondary selection from TGLViewer.
/// Just calls SecSelected(track) in model object which emits a signal.
/// This is used in user code for alternate selection of good / bad tracks.

void TEveTrackGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   if (gDebug > 0)
   {
      printf("TEveTrackGL::ProcessSelection %d names on the stack (z1=%g, z2=%g).\n",
             rec.GetN(), rec.GetMinZ(), rec.GetMaxZ());
      printf("  Names: ");
      for (Int_t j=0; j<rec.GetN(); ++j) printf ("%d ", rec.GetItem(j));
      printf("\n");
   }

   fTrack->SecSelected(fTrack);
}

////////////////////////////////////////////////////////////////////////////////
/// GL rendering code.
/// Virtual from TGLLogicalShape.

void TEveTrackGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   TEveLineGL::DirectDraw(rnrCtx);

   RenderPathMarksAndFirstVertex(rnrCtx);
}

////////////////////////////////////////////////////////////////////////////////
/// Render path-marks and the first vertex, if required.

void TEveTrackGL::RenderPathMarksAndFirstVertex(TGLRnrCtx& rnrCtx) const
{
   TEveTrackPropagator          &rTP = *fTrack->GetPropagator();
   const TEveTrack::vPathMark_t &pms =  fTrack->RefPathMarks();
   if ( ! pms.empty())
   {
      Float_t *pnts = new Float_t[3*pms.size()]; // maximum
      Int_t    cnt  = 0;
      Int_t    n    = 0;
      for (Int_t i = 0; i < fTrack->fLastPMIdx; ++i)
      {
         const TEvePathMarkD &pm = pms[i];
         if ((pm.fType == TEvePathMarkD::kDaughter  && rTP.GetRnrDaughters())  ||
             (pm.fType == TEvePathMarkD::kReference && rTP.GetRnrReferences()) ||
             (pm.fType == TEvePathMarkD::kDecay     && rTP.GetRnrDecay())      ||
             (pm.fType == TEvePathMarkD::kCluster2D && rTP.GetRnrCluster2Ds()))
         {
            pnts[n  ] = pm.fV.fX;
            pnts[n+1] = pm.fV.fY;
            pnts[n+2] = pm.fV.fZ;
            n += 3;
            ++cnt;
         }
      }
      TGLUtil::RenderPolyMarkers(rTP.RefPMAtt(), 0, pnts, cnt,
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
      delete [] pnts;
   }

   // fist vertex
   if (rTP.GetRnrFV() && fTrack->GetLastPoint())
      TGLUtil::RenderPolyMarkers(rTP.RefFVAtt(), 0, fTrack->GetP(), 1,
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
}
