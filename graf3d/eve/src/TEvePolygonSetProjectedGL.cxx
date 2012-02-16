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

   // fDLCache = kFALSE; // Disable DL.
   fMultiColor = kTRUE; // Potentially false, reset in DirectDraw().
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEvePolygonSetProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEvePolygonSetProjected>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::SetBBox()
{
   // Setup bounding-box information.

   SetAxisAlignedBBox(fM->AssertBBox());
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

   if (fM->fPols.size() == 0) return;

   Bool_t done_p = kFALSE;

   if (fM->GetMiniFrame())
   {
      std::map<Edge_t, Int_t> edges;

      for (TEvePolygonSetProjected::vpPolygon_ci i = fM->fPols.begin();
           i != fM->fPols.end(); ++i)
      {
         for(Int_t k = 0; k < i->fNPnts - 1; ++k)
         {
            ++edges[Edge_t(i->fPnts[k], i->fPnts[k+1])];
         }
         ++edges[Edge_t(i->fPnts[0], i->fPnts[i->fNPnts - 1])];
      }

      glBegin(GL_LINES);
      for (std::map<Edge_t, Int_t>::iterator i = edges.begin(); i != edges.end(); ++i)
      {
         if (i->second == 1)
         {
            glVertex3fv(fM->fPnts[i->first.fI].Arr());
            glVertex3fv(fM->fPnts[i->first.fJ].Arr());
            done_p = kTRUE;
         }
      }
      glEnd();
   }

   if ( ! done_p)
   {
      for (TEvePolygonSetProjected::vpPolygon_ci i = fM->fPols.begin();
           i != fM->fPols.end(); ++i)
      {
         glBegin(GL_LINE_LOOP);
         for(Int_t k = 0; k < i->fNPnts; ++k)
         {
            glVertex3fv(fM->fPnts[i->fPnts[k]].Arr());
         }
         glEnd();
      }
   }
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Do GL rendering.

   if (fM->fPols.size() == 0) return;

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);

   glDisable(GL_LIGHTING);
   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);

   // This tells TGLObject we don't want display-lists in some cases.
   fMultiColor = fM->fDrawFrame;

   // polygons
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1, 1);
   GLUtesselator *tessObj = TGLUtil::GetDrawTesselator3fv();

   TEveVector* pnts = fM->fPnts;
   for (TEvePolygonSetProjected::vpPolygon_ci i = fM->fPols.begin();
        i != fM->fPols.end(); ++i)
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
         glNormal3f(0, 0, 1);
         Double_t coords[3];
         coords[2] = 0;
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
   if (fM->fDrawFrame)
   {
      TGLUtil::Color(fM->fLineColor);
      glEnable(GL_LINE_SMOOTH);
      TGLUtil::LineWidth(fM->fLineWidth);
      DrawOutline();
   }

   glPopAttrib();
}

//______________________________________________________________________________
void TEvePolygonSetProjectedGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl) const
{
   // Draw polygons in highlight mode.

   if (lvl < 0) lvl = pshp->GetSelected();

   glColor4ubv(rnrCtx.ColorSet().Selection(lvl).CArr());
   TGLUtil::LockColor();

   if (fM->GetHighlightFrame())
   {
      DrawOutline();
   }
   else
   {
      Draw(rnrCtx);
   }

   TGLUtil::UnlockColor();
}
