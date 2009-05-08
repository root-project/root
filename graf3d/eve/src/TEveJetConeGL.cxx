// @(#)root/eve:$Id$
// Author: Matevz Tadel, Jochen Thaeder 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveJetConeGL.h"
#include "TEveJetCone.h"

#include "TMath.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

//______________________________________________________________________________
// OpenGL renderer class for TEveJetCone.
//

ClassImp(TEveJetConeGL);

//______________________________________________________________________________
TEveJetConeGL::TEveJetConeGL() :
   TGLObject(), fM(0)
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveJetConeGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (SetModelCheckClass(obj, TEveJetCone::Class())) {
      fM = dynamic_cast<TEveJetCone*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveJetConeGL::SetBBox()
{
   // Set bounding box.

   // !! This ok if master sub-classed from TAttBBox
   SetAxisAlignedBBox(((TEveJetCone*)fExternalObj)->AssertBBox());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveJetConeGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Render with OpenGL.

   // printf("TEveJetConeGL::DirectDraw LOD %d\n", rnrCtx.CombiLOD());

   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);

   glDisable(GL_CULL_FACE);
   glEnable(GL_NORMALIZE);

   glBegin(GL_TRIANGLE_FAN);
   glVertex3fv(fM->fApex);
   if ( fM->fBasePoints.size() > 2)
   {
      TEveJetCone::vTEveVector_ci prev = fM->fBasePoints.end(); --prev;
      TEveJetCone::vTEveVector_ci i    = fM->fBasePoints.begin();
      TEveJetCone::vTEveVector_ci next = i; ++next;

      TEveVector norm_buf;
      TEveVector beg_normal = TMath::Cross((*i - fM->fApex).Arr(), (*next - *prev).Arr(), norm_buf.Arr());

      glNormal3fv(beg_normal);
      glVertex3fv(fM->fBasePoints.front());

      prev = i;  i = next;  ++next;

      while (i != fM->fBasePoints.begin())
      {
         glNormal3fv(TMath::Cross((*i - fM->fApex).Arr(), (*next - *prev).Arr(), norm_buf.Arr()));
         glVertex3fv(*i);

         prev = i;
         i    = next;
         ++next; if (next == fM->fBasePoints.end()) next = fM->fBasePoints.begin();
      }

      glNormal3fv(beg_normal);
      glVertex3fv(fM->fBasePoints.front());
   }
   glEnd();

   glPopAttrib();
}
