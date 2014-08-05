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

   TEveTrackGL::SetModel(obj);
   fM = DynCast<TEveTrackProjected>(obj);
   return kTRUE;
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
      TGLCapabilityEnabler sw_smooth(GL_LINE_SMOOTH, fM->fSmooth);
      TGLCapabilityEnabler sw_blend(GL_BLEND, fM->fSmooth);
      Int_t start = 0;
      Float_t* p  = fM->GetP();
      TGLUtil::LockColor(); // Keep color from TGLPhysicalShape.
      for (std::vector<Int_t>::iterator bpi = fM->fBreakPoints.begin();
           bpi != fM->fBreakPoints.end(); ++bpi)
      {
         Int_t size = *bpi - start;
         TGLUtil::RenderPolyLine(*fM, fM->GetMainTransparency(), p, size);
         p     += 3*size;
         start +=   size;
      }
      TGLUtil::UnlockColor();
   }

   // markers on lines
   if (fM->fRnrPoints)
   {
      TGLUtil::RenderPolyMarkers(*fM, 0,
                                 fM->GetP(), fM->Size(),
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
   }

   // break-points
   if (fM->fBreakPoints.size() > 1 && fM->fPropagator->GetRnrPTBMarkers())
   {
      // Last break-point is last point on track, do not draw it.
      Int_t  nbp   = fM->fBreakPoints.size() - 1;
      Bool_t bmb   = fM->fPropagator->GetProjTrackBreaking() == TEveTrackPropagator::kPTB_Break;
      Int_t  nbptd = bmb ? 2*nbp : nbp;
      std::vector<Float_t> pnts(3*nbptd);
      Int_t n = 0;
      for (Int_t i = 0; i < nbp; ++i, n+=3)
      {
         fM->GetPoint(fM->fBreakPoints[i] - 1, pnts[n], pnts[n+1], pnts[n+2]);
         if (bmb)
         {
            n += 3;
            fM->GetPoint(fM->fBreakPoints[i], pnts[n], pnts[n+1], pnts[n+2]);
         }
      }
      TGLUtil::RenderPolyMarkers(fM->fPropagator->RefPTBAtt(), 0,
                                 &pnts[0], nbptd,
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
   }

   RenderPathMarksAndFirstVertex(rnrCtx);
}
