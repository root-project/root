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

   const TEveJetCone::vTEveVector_t &BP = fM->fBasePoints;
   const Int_t                       NP = BP.size();

   if (NP > 2)
   {
      glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LIGHTING_BIT);

      glDisable(GL_CULL_FACE);
      glEnable(GL_NORMALIZE);
      Int_t lmts = 1;
      glLightModeliv(GL_LIGHT_MODEL_TWO_SIDE, &lmts);

      Int_t prev = NP - 1;
      Int_t i    = 0;
      Int_t next = 1;

      TEveVector curr_normal;
      TEveVector prev_normal;
      TMath::Cross((BP[next] - BP[prev]).Arr(), (BP[i] - fM->fApex).Arr(), prev_normal.Arr());

      prev = i; i = next; ++next;

      glBegin(GL_TRIANGLES);
      do
      {
         TMath::Cross((BP[next] - BP[prev]).Arr(), (BP[i] - fM->fApex).Arr(), curr_normal.Arr());

         glNormal3fv(prev_normal);
         glVertex3fv(BP[prev]);

         glNormal3fv(prev_normal + curr_normal);
         glVertex3fv(fM->fApex);

         glNormal3fv(curr_normal);
         glVertex3fv(BP[i]);
         
         prev_normal = curr_normal;

         prev = i;
         i    = next;
         ++next; if (next >= NP) next = 0;
      } while (prev != 0);
      glEnd();

      glPopAttrib();
   }
}
