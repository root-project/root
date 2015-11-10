// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLCameraGuide.h"
#include "TGLRnrCtx.h"
#include "TGLCamera.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"
#include "TGLSelectRecord.h"

#include "TMath.h"

/** \class TGLCameraGuide
\ingroup opengl
Draws arrows showing camera orientation in the overlay.
X, Y position is in range 0, 1.
*/

ClassImp(TGLCameraGuide);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLCameraGuide::TGLCameraGuide(Float_t x, Float_t y, Float_t s,
                               ERole role, EState state) :
   TGLOverlayElement(role, state),
   fXPos(x), fYPos(y), fSize(s),
   fSelAxis(-1), fInDrag(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has entered overlay area.

Bool_t TGLCameraGuide::MouseEnter(TGLOvlSelectRecord& /*rec*/)
{
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle overlay event.
/// Return TRUE if event was handled.

Bool_t TGLCameraGuide::Handle(TGLRnrCtx&          rnrCtx,
                              TGLOvlSelectRecord&  selRec,
                              Event_t*             event)
{
   if (selRec.GetN() < 2) return kFALSE;
   Int_t recID = selRec.GetItem(1);

   if (recID == 4)
      fSelAxis = 4;
   else
      fSelAxis = 0;

   switch (event->fType)
   {
      case kButtonPress:
      {
         if (recID == 4)
            fInDrag = kTRUE;
         return kTRUE;
      }
      case kButtonRelease:
      {
         fInDrag = kFALSE;
         return kTRUE;
      }
      case kMotionNotify:
      {
         if (fInDrag)
         {
            const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
            if (vp.Width() == 0 || vp.Height() == 0) return kFALSE;

            fXPos = TMath::Range(0.0f, 1.0f, (Float_t)(event->fX) / vp.Width());
            fYPos = TMath::Range(0.0f, 1.0f, 1.0f - (Float_t)(event->fY) / vp.Height());
         }
         return kTRUE;
      }
      default:
      {
         return kFALSE;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has left overlay area.

void TGLCameraGuide::MouseLeave()
{
   fSelAxis = -1;
   fInDrag  = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Render the camera axis arrows.

void TGLCameraGuide::Render(TGLRnrCtx& rnrCtx)
{
   TGLCapabilitySwitch lgt_off(GL_LIGHTING, kFALSE);

   rnrCtx.ProjectionMatrixPushIdentity();
   glPushMatrix();
   glLoadIdentity();
   glTranslatef(-1.0f, -1.0f, 0.0f);
   glScalef(2.0f, 2.0f, -2.0f);
   glTranslatef(fXPos, fYPos, -0.25f);

   Float_t aspect= rnrCtx.RefCamera().RefViewport().Aspect();
   if (aspect > 1)
      glScalef(1.0f / aspect, 1.0f, 1.0f);
   else if (aspect < 1)
      glScalef(1.0f, aspect, 1.0f);

   Float_t dr[2];
   glGetFloatv(GL_DEPTH_RANGE, dr);
   glDepthRange(0, 0.01);

   TGLVertex3 c;
   TGLVector3 e;
   const TGLMatrix &mv = rnrCtx.RefCamera().RefModelViewMatrix();

   glPushName(1);
   mv.GetBaseVec(1, e);
   e *= fSize;
   TGLUtil::DrawLine(c, e, TGLUtil::kLineHeadArrow, 0.1*fSize,
                     fSelAxis == 1 ? TGLUtil::fgYellow : TGLUtil::fgRed);

   glLoadName(2);
   mv.GetBaseVec(2, e);
   e *= fSize;
   TGLUtil::DrawLine(c, e, TGLUtil::kLineHeadArrow, 0.1*fSize,
                     fSelAxis == 2 ? TGLUtil::fgYellow : TGLUtil::fgGreen);

   glLoadName(3);
   mv.GetBaseVec(3, e);
   e *= fSize;
   TGLUtil::DrawLine(c, e, TGLUtil::kLineHeadArrow, 0.1*fSize,
                     fSelAxis == 3 ? TGLUtil::fgYellow : TGLUtil::fgBlue);

   glLoadName(4);
   TGLUtil::DrawSphere(c, 0.08*fSize,
                       fSelAxis == 4 ? TGLUtil::fgYellow : rnrCtx.ColorSet().Foreground().CArr());

   glPopName();

   glDepthRange(dr[0], dr[1]);

   glPopMatrix();
   rnrCtx.ProjectionMatrixPop();
}
