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

//==============================================================================
// TEveTrackGL
//==============================================================================

//______________________________________________________________________________
//
// GL-renderer for TEveTrack class.

ClassImp(TEveTrackGL);

//______________________________________________________________________________
TEveTrackGL::TEveTrackGL() : TEveLineGL()
{
   // Default constructor.

   // fDLCache = false; // Disable display list.
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveTrackGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (TEveLineGL::SetModel(obj) == kFALSE) return kFALSE;
   if (SetModelCheckClass(obj, TEveTrack::Class())) {
      fTrack = dynamic_cast<TEveTrack*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveTrackGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes secondary selection from TGLViewer.
   // Calls TPointSet3D::PointSelected(Int_t) with index of selected
   // point as an argument.

   printf("TEveTrackGL::ProcessSelection %d names on the stack (z1=%g, z2=%g).\n",
          rec.GetN(), rec.GetMinZ(), rec.GetMaxZ());
   printf("  Names: ");
   for (Int_t j=0; j<rec.GetN(); ++j) printf ("%d ", rec.GetItem(j));
   printf("\n");

   ((TEveTrack*)fM)->SecSelected((TEveTrack*)fM);
}

//______________________________________________________________________________
void TEveTrackGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // GL rendering code.
   // Virtual from TGLLogicalShape.

   TEveLineGL::DirectDraw(rnrCtx);

   // path-marks
   const TEveTrack::vPathMark_t& pms = fTrack->RefPathMarks();
   TEveTrackPropagator& rTP = *fTrack->GetPropagator();
   if (pms.size())
   {
      Float_t* pnts = new Float_t[3*pms.size()]; // maximum
      Int_t n = 0;
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
               pnts[3*n  ] = pm->fV.fX;
               pnts[3*n+1] = pm->fV.fY;
               pnts[3*n+2] = pm->fV.fZ;
               ++n;
            }
         }
      }
      TGLUtil::RenderPolyMarkers(rTP.RefPMAtt(), pnts, n,
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
