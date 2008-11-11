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
#include "TEveVSDStructs.h"

#include "TGLRnrCtx.h"
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
void TEvePolygonSetProjectedGL::DirectDraw(TGLRnrCtx & /*rnrCtx*/) const
{
   // Do GL rendering.

   TEvePolygonSetProjected& refPS = * (TEvePolygonSetProjected*) fExternalObj;
   if (refPS.fPols.size() == 0) return;

   fMultiColor = (refPS.fFillColor != refPS.fLineColor);

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);

   glDisable(GL_LIGHTING);
   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);

   // polygons
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.,1.);
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

   // outline
   TGLUtil::Color(refPS.fLineColor);
   glEnable(GL_LINE_SMOOTH);

   glLineWidth(refPS.fLineWidth);
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

   glPopAttrib();
}
