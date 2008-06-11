// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTrackProjectedGL.h"
#include "TEveTrackProjected.h"
#include "TEveTrackPropagator.h"
#include "TEveProjectionManager.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"

//==============================================================================
// TEveTrackProjectedGL
//==============================================================================

//______________________________________________________________________________
//
// GL-renderer for TEveTrackProjected class.

ClassImp(TEveTrackProjectedGL);

//______________________________________________________________________________
TEveTrackProjectedGL::TEveTrackProjectedGL() : TEveTrackGL(), fM(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveTrackProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (TEveTrackGL::SetModel(obj) == kFALSE) return kFALSE;
   if (SetModelCheckClass(obj, TEveTrackProjected::Class())) {
      fM = dynamic_cast<TEveTrackProjected*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjectedGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Draw track with GL.

   // printf("TEveTrackProjectedGL::DirectDraw Style %d, LOD %d\n", flags.Style(), flags.LOD());
   if (fM->Size() == 0)
      return;

   // lines
   if (fM->fRnrLine)
   {
      TGLCapabilitySwitch sw_smooth(GL_LINE_SMOOTH, fM->fSmooth);
      TGLCapabilitySwitch sw_blend(GL_BLEND, fM->fSmooth);
      Int_t start = 0;
      Float_t* p  = fM->GetP();
      TGLUtil::LockColor(); // Keep color from TGLPhysicalShape.
      for (std::vector<Int_t>::iterator bpi = fM->fBreakPoints.begin();
           bpi != fM->fBreakPoints.end(); ++bpi)
      {
         Int_t size = *bpi - start;
         TGLUtil::RenderPolyLine(*fM, p, size);
         p     += 3*size;
         start +=   size;
      }
      TGLUtil::UnlockColor();
   }

   // markers on lines
   if (fM->fRnrPoints)
   {
      TGLUtil::RenderPolyMarkers(*fM, fM->GetP(), fM->Size(),
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
   }

   // path-marks
   const TEveTrack::vPathMark_t& pms = fTrack->RefPathMarks();
   TEveTrackPropagator& rTP = *fM->GetPropagator();
   if (pms.size())
   {
      Float_t* pnts = new Float_t[3*pms.size()]; // maximum
      Float_t*  pnt = pnts;
      Int_t   pntsN = 0;
      Bool_t accept;
      for (TEveTrack::vPathMark_ci pm = pms.begin(); pm != pms.end(); ++pm)
      {
         accept = kFALSE;
         switch (pm->fType)
         {
            case TEvePathMark::kDaughter:
               if (rTP.GetRnrDaughters())  accept = kTRUE;
               break;
            case TEvePathMark::kReference:
               if (rTP.GetRnrReferences()) accept = kTRUE;
               break;
            case TEvePathMark::kDecay:
               if (rTP.GetRnrDecay())      accept = kTRUE;
               break;
            case TEvePathMark::kCluster2D:
               if (rTP.GetRnrCluster2Ds()) accept = kTRUE;
               break;
         }
         if (accept)
         {
            if ((TMath::Abs(pm->fV.fZ) < rTP.GetMaxZ()) && (pm->fV.Perp() < rTP.GetMaxR()))
            {
               pnt[0] = pm->fV.fX;
               pnt[1] = pm->fV.fY;
               pnt[2] = pm->fV.fZ;
               fM->fProjection->ProjectPointFv(pnt);
               pnt   += 3;
               ++pntsN;
            }
         }
      }
      TGLUtil::RenderPolyMarkers(rTP.RefPMAtt(), pnts, pntsN,
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
      delete [] pnts;
   }

   // fist vertex
   if (rTP.GetRnrFV() && fTrack->GetLastPoint())
      TGLUtil::RenderPolyMarkers(rTP.RefFVAtt(), fTrack->GetP(), 1,
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
}
