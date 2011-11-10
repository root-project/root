// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveStraightLineSetGL.h"
#include "TEveStraightLineSet.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"

//==============================================================================
//==============================================================================
// TEveStraightLineSetGL
//==============================================================================

//______________________________________________________________________________
//
// GL-renderer for TEveStraightLineSet class.

ClassImp(TEveStraightLineSetGL);

//______________________________________________________________________________
TEveStraightLineSetGL::TEveStraightLineSetGL() : TGLObject(), fM(0)
{
   // Constructor.

   // fDLCache = false; // Disable display list.
   fMultiColor = kTRUE;
}

//==============================================================================

//______________________________________________________________________________
Bool_t TEveStraightLineSetGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveStraightLineSet>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void TEveStraightLineSetGL::SetBBox()
{
   // Setup bounding box information.

   SetAxisAlignedBBox(((TEveStraightLineSet*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
Bool_t TEveStraightLineSetGL::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   // Override from TGLObject.
   // To account for large point-sizes we modify the projection matrix
   // during selection and thus we need a direct draw.

   if (rnrCtx.Selection()) return kFALSE;
   return TGLObject::ShouldDLCache(rnrCtx);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw function for TEveStraightLineSetGL. Skips line-pass of outline mode.

   if (rnrCtx.IsDrawPassOutlineLine())
      return;

   TGLObject::Draw(rnrCtx);
}

//______________________________________________________________________________
void TEveStraightLineSetGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Render the line-set with GL.

   // printf("TEveStraightLineSetGL::DirectDraw LOD %d\n", rnrCtx.ShapeLOD());

   TEveStraightLineSet& mL = * fM;

   // set depth range when selection is disabled, else can't pick camera center
   if (mL.GetDepthTest() == kFALSE && rnrCtx.Selection() == kFALSE)
   {
      glPushAttrib(GL_VIEWPORT_BIT);
      glDepthRange(0, 0.1); 
   }

   // lines
   if (mL.GetRnrLines() && mL.GetLinePlex().Size() > 0)
   {
      glPushAttrib(GL_LINE_BIT | GL_ENABLE_BIT);
      glDisable(GL_LIGHTING);
      TGLUtil::LineWidth(mL.GetLineWidth());
      if (mL.GetLineStyle() > 1) {
         // Int_t    fac = 1;
         UShort_t pat = 0xffff;
         switch (mL.GetLineStyle()) {
            case 2:  pat = 0x3333; break;
            case 3:  pat = 0x5555; break;
            case 4:  pat = 0xf040; break;
            case 5:  pat = 0xf4f4; break;
            case 6:  pat = 0xf111; break;
            case 7:  pat = 0xf0f0; break;
            case 8:  pat = 0xff11; break;
            case 9:  pat = 0x3fff; break;
            case 10: pat = 0x08ff; /* fac = 2; */ break;
         }
         glLineStipple(1, pat);
         glEnable(GL_LINE_STIPPLE);
      }

      // During selection extend picking region for large line-widths.
      Bool_t changePM = rnrCtx.Selection() && mL.GetLineWidth() > rnrCtx.GetPickRadius();
      if (changePM)
         TGLUtil::BeginExtendPickRegion((Float_t) rnrCtx.GetPickRadius() / mL.GetLineWidth());

      TEveChunkManager::iterator li(mL.GetLinePlex());
      if (rnrCtx.SecSelection())
      {
         GLuint name = 0;
         glPushName(1);
         glPushName(0);
         while (li.next())
         {
            TEveStraightLineSet::Line_t& l = * (TEveStraightLineSet::Line_t*) li();
            glLoadName(l.fId);
            {
               glBegin(GL_LINES);
               glVertex3f(l.fV1[0], l.fV1[1], l.fV1[2]);
               glVertex3f(l.fV2[0], l.fV2[1], l.fV2[2]);
               glEnd();
            }
            name ++;
         }
         glPopName();
         glPopName();
      }
      else
      {
         glBegin(GL_LINES);
         while (li.next())
         {
            TEveStraightLineSet::Line_t& l = * (TEveStraightLineSet::Line_t*) li();
            glVertex3f(l.fV1[0], l.fV1[1], l.fV1[2]);
            glVertex3f(l.fV2[0], l.fV2[1], l.fV2[2]);
         }
         glEnd();
      }

      if (changePM)
         TGLUtil::EndExtendPickRegion();

      glPopAttrib();
   }


   // markers
   if (mL.GetRnrMarkers() && mL.GetMarkerPlex().Size() > 0)
   {
      TEveChunkManager::iterator mi(mL.GetMarkerPlex());
      Float_t* pnts = new Float_t[mL.GetMarkerPlex().Size()*3];
      Float_t* pnt  = pnts;
      while (mi.next())
      {
         TEveStraightLineSet::Marker_t& m = * (TEveStraightLineSet::Marker_t*) mi();
         pnt[0] = m.fV[0];
         pnt[1] = m.fV[1];
         pnt[2] = m.fV[2];
         pnt   += 3;
      }
      if (rnrCtx.SecSelection()) glPushName(2);
      TGLUtil::RenderPolyMarkers((TAttMarker&)mL, mL.GetMainTransparency(),
                                 pnts, mL.GetMarkerPlex().Size(),
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection(),
                                 rnrCtx.SecSelection());
      if (rnrCtx.SecSelection()) glPopName();
      delete [] pnts;
   }

   if (mL.GetDepthTest() == kFALSE && rnrCtx.Selection() == kFALSE)
      glPopAttrib();
}

//==============================================================================

//______________________________________________________________________________
void TEveStraightLineSetGL::ProcessSelection(TGLRnrCtx& /*rnrCtx*/,
                                             TGLSelectRecord& rec)
{
   // Process results of the secondary selection.

   if (rec.GetN() != 3) return;
   if (rec.GetItem(1) == 1)
   {
      printf("selected line %d\n", rec.GetItem(2));
   }
   else
   {
      TEveStraightLineSet::Marker_t& m = * (TEveStraightLineSet::Marker_t*) fM->GetMarkerPlex().Atom(rec.GetItem(2));
      printf("Selected point %d on line %d\n", rec.GetItem(2), m.fLineId);
   }
}
