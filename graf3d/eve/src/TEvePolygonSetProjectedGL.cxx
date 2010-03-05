// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePolygonSetProjectedGL.h"
#include "TEvePolygonSetProjected.h"
#include "TEveVector.h"

#include "TGLRnrCtx.h"
#include "TGLCamera.h"
#include "TGLPhysicalShape.h"
#include "TGLIncludes.h"

//==============================================================================
//==============================================================================
// TEvePolygonSetProjectedGL
//==============================================================================

//______________________________________________________________________________
//
// GL-renderer for TEvePolygonSetProjected class.

ClassImp(TEvePolygonSetProjectedGL);

//______________________________________________________________________________
TEvePolygonSetProjectedGL::TEvePolygonSetProjectedGL() : TGLObject()
{
   // Constructor

   // fDLCache = false; // Disable DL.
   fMultiColor = kTRUE; // Potentially false, reset in DirectDraw().
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEvePolygonSetProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   return SetModelCheckClass(obj, TEvePolygonSetProjected::Class());
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::SetBBox()
{
   // Setup bounding-box information.

   SetAxisAlignedBBox(((TEvePolygonSetProjected*)fExternalObj)->AssertBBox());
}

/******************************************************************************/

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw function for TEvePolygonSetProjectedGL.
   // Skips line-pass of outline mode.

   if (rnrCtx.IsDrawPassOutlineLine())
      return;

   TGLObject::Draw(rnrCtx);
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::DrawOutline() const
{
   // Draw polygons outline.

   TEvePolygonSetProjected& refPS = * (TEvePolygonSetProjected*) fExternalObj;
   if (refPS.fPols.size() == 0) return;

   Int_t vi;
   for (TEvePolygonSetProjected::vpPolygon_ci i = refPS.fPols.begin();
        i != refPS.fPols.end(); ++i)
   {
      glBegin(GL_LINE_LOOP);
      for(Int_t k = 0; k < (*i).fNPnts; ++k)
      {
         vi = (*i).fPnts[k];
         glVertex3fv(refPS.fPnts[vi].Arr());
      }
      glEnd();
   }
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Do GL rendering.

   TEvePolygonSetProjected& refPS = * (TEvePolygonSetProjected*) fExternalObj;
   if (refPS.fPols.size() == 0) return;

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);

   glDisable(GL_LIGHTING);
   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);

   fMultiColor = (refPS.fDrawFrame && refPS.fFillColor != refPS.fLineColor);

   // polygons
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.0f,1.0f);
   GLUtesselator *tessObj = TGLUtil::GetDrawTesselator3fv();

   TEveVector* pnts = refPS.fPnts;
   for (TEvePolygonSetProjected::vpPolygon_ci i = refPS.fPols.begin();
        i != refPS.fPols.end(); ++i)
   {
      Int_t vi; //current vertex index of curent polygon
      Int_t pntsN = (*i).fNPnts; // number of points in current polygon
      if (pntsN < 4)
      {
         glBegin(GL_POLYGON);
         for (Int_t k = 0; k < pntsN; ++k)
         {
            vi = (*i).fPnts[k];
            glVertex3fv(pnts[vi].Arr());
         }
         glEnd();
      }
      else
      {
         gluBeginPolygon(tessObj);
         gluNextContour(tessObj, (GLenum)GLU_UNKNOWN);
         glNormal3f(0., 0., 1.);
         Double_t coords[3];
         coords[2] = 0.;
         for (Int_t k = 0; k < pntsN; ++k)
         {
            vi = (*i).fPnts[k];
            coords[0] = pnts[vi].fX;
            coords[1] = pnts[vi].fY;
            gluTessVertex(tessObj, coords, pnts[vi].Arr());
         }
         gluEndPolygon(tessObj);
      }
   }
   glDisable(GL_POLYGON_OFFSET_FILL);

   // Outline
   if (refPS.fDrawFrame)
   {
      TGLUtil::Color(refPS.fLineColor);
      glEnable(GL_LINE_SMOOTH);
      TGLUtil::LineWidth(refPS.fLineWidth);
      DrawOutline();
   }

   glPopAttrib();
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp) const
{
   // Draw polygons in highlight mode.

   TEvePolygonSetProjected& refPS = * (TEvePolygonSetProjected*) fExternalObj;

   if (refPS.GetHighlightFrame())
   {
      glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
      glDisable(GL_LIGHTING);
      glEnable(GL_LINE_SMOOTH);

      glColor4ubv(rnrCtx.ColorSet().Selection(pshp->GetSelected()).CArr());

      const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
      Int_t inner[4][2] = { { 0,-1}, { 1, 0}, { 0, 1}, {-1, 0} };
      Int_t outer[8][2] = { {-1,-1}, { 1,-1}, { 1, 1}, {-1, 1},
                            { 0,-2}, { 2, 0}, { 0, 2}, {-2, 0} };

      rnrCtx.SetHighlightOutline(kTRUE);
      TGLUtil::LockColor();
      Int_t first_outer = (rnrCtx.CombiLOD() == TGLRnrCtx::kLODHigh) ? 0 : 4;
      for (int i = first_outer; i < 8; ++i)
      {
         glViewport(vp.X() + outer[i][0], vp.Y() + outer[i][1], vp.Width(), vp.Height());
         DrawOutline();
      }
      TGLUtil::UnlockColor();
      rnrCtx.SetHighlightOutline(kFALSE);

      TGLUtil::Color(refPS.fLineColor);
      for (int i = 0; i < 4; ++i)
      {
         glViewport(vp.X() + inner[i][0], vp.Y() + inner[i][1], vp.Width(), vp.Height());
         DrawOutline();
      }
      glViewport(vp.X(), vp.Y(), vp.Width(), vp.Height());

      pshp->SetupGLColors(rnrCtx);
      Float_t dr[2];
      glGetFloatv(GL_DEPTH_RANGE,dr);
      glDepthRange(dr[0], 0.5*dr[1]);
      DrawOutline();
      glDepthRange(dr[0], dr[1]);

      glPopAttrib();
   }
   else
   {
      TGLLogicalShape::DrawHighlight(rnrCtx, pshp);
   }
}
