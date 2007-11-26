// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveTrackProjectedGL.h>
#include <TEveTrackProjected.h>
#include <TEveTrackPropagator.h>
#include <TEveProjectionManager.h>
#include <TEveGLUtil.h>

#include <TGLRnrCtx.h>
#include <TGLIncludes.h>

//______________________________________________________________________________
// TEveTrackProjectedGL
//
// GL-renderer for TEveTrackProjected class.

ClassImp(TEveTrackProjectedGL)

//______________________________________________________________________________
TEveTrackProjectedGL::TEveTrackProjectedGL() : TEveTrackGL(), fM(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
TEveTrackProjectedGL::~TEveTrackProjectedGL()
{}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveTrackProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   if(TEveTrackGL::SetModel(obj) == kFALSE) return kFALSE;
   if(SetModelCheckClass(obj, TEveTrackProjected::Class())) {
      fM = dynamic_cast<TEveTrackProjected*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackProjectedGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // printf("TEveTrackProjectedGL::DirectDraw Style %d, LOD %d\n", flags.Style(), flags.LOD());
   if (rnrCtx.DrawPass() == TGLRnrCtx::kPassOutlineLine || fM->Size() == 0)
      return;

   // lines
   Int_t start = 0;
   Float_t* p = fM->GetP();
   for (std::vector<Int_t>::iterator bpi = fM->fBreakPoints.begin();
        bpi != fM->fBreakPoints.end(); ++bpi)
   {
      Int_t size = *bpi - start;
      if (fM->fRnrLine)   TEveGLUtil::RenderLine(*fM, p, size);
      if (fM->fRnrPoints) TEveGLUtil::RenderPolyMarkers(*fM, p, size);
      p     += 3*size;
      start +=   size;
   }

   // path-marks
   std::vector<TEvePathMark*>& pm = fM->fPathMarks;
   TEveTrackPropagator& RS = *fM->GetPropagator();
   if(pm.size())
   {
      Float_t* pnts = new Float_t[3*pm.size()]; // maximum
      Int_t N = 0;
      Bool_t accept;
      for(std::vector<TEvePathMark*>::iterator i=pm.begin(); i!=pm.end(); ++i)
      {
         accept = kFALSE;
         switch((*i)->type)
         {
            case(TEvePathMark::Daughter):
               if(RS.fRnrDaughters) accept = kTRUE;
               break;
            case(TEvePathMark::Reference):
               if(RS.fRnrReferences) accept = kTRUE;
               break;
            case(TEvePathMark::Decay):
               if(RS.fRnrDecay) accept = kTRUE;
               break;
         }
         if(accept)
         {
            if((TMath::Abs((*i)->V.z) < RS.fMaxZ) && ((*i)->V.Perp() < RS.fMaxR))
            {
               pnts[3*N  ] =(*i)->V.x;
               pnts[3*N+1] =(*i)->V.y;
               pnts[3*N+2] =(*i)->V.z;
               fM->fProjection->ProjectPoint(pnts[3*N  ], pnts[3*N+1], pnts[3*N+2]);
               N++;
            }
         }
      }
      TEveGLUtil::RenderPolyMarkers(RS.fPMAtt, pnts, N);
      delete [] pnts;
   }

   // fist vertex
   if(RS.fRnrFV && fTrack->GetLastPoint())
      TEveGLUtil::RenderPolyMarkers(RS.fFVAtt, fTrack->GetP(), 1);
}
