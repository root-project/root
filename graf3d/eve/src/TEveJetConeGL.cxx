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
#include "TEveProjectionManager.h"

#include "TMath.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

//==============================================================================
// TEveJetConeGL
//==============================================================================

//______________________________________________________________________________
// OpenGL renderer class for TEveJetCone.
//

ClassImp(TEveJetConeGL);

//______________________________________________________________________________
TEveJetConeGL::TEveJetConeGL() :
   TGLObject(), fC(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveJetConeGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fC = SetModelDynCast<TEveJetCone>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void TEveJetConeGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveJetCone*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TEveJetConeGL::DLCacheClear()
{
   // Clear DL cache and reset internal point array.

   fP.clear();
   TGLObject::DLCacheClear();
}

//______________________________________________________________________________
void TEveJetConeGL::CalculatePoints() const
{
   // Calculate points for drawing.

   assert(fC->fNDiv > 2);

   const Int_t  NP = fC->fNDiv;
   fP.resize(NP);
   {
      Float_t angle_step = TMath::TwoPi() / NP;
      Float_t angle      = 0;
      for (Int_t i = 0; i < NP; ++i, angle += angle_step)
      {
         fP[i] = fC->CalcBaseVec(angle);
      }
   }
}

//______________________________________________________________________________
void TEveJetConeGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw the cone.

   if (fP.empty()) CalculatePoints();

   if (fC->fHighlightFrame && rnrCtx.Highlight())
   {
      glPushAttrib(GL_ENABLE_BIT);
      glDisable(GL_LIGHTING);

      if (fC->fDrawFrame)
      {
         TGLUtil::LineWidth(fC->fLineWidth);
         TGLUtil::Color(fC->fLineColor);
      }

      const Int_t NP = fP.size();
      glBegin(GL_LINE_LOOP);
      for (Int_t i = 0; i < NP; ++i)
         glVertex3fv(fP[i]);
      glEnd();
      glBegin(GL_LINES);
      Double_t angle = 0;
      for (Int_t i = 0; i < 4; ++i, angle += TMath::PiOver2())
      {
         glVertex3fv(fC->fApex);
         glVertex3fv(fC->CalcBaseVec(angle));
      }
      glEnd();

      glPopAttrib();
   }
   else
   {
      TGLObject::Draw(rnrCtx);
   }
}

//______________________________________________________________________________
void TEveJetConeGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Render with OpenGL.

   // printf("TEveJetConeGL::DirectDraw LOD %d\n", rnrCtx.CombiLOD());

   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LIGHTING_BIT);

   glDisable(GL_CULL_FACE);
   glEnable(GL_NORMALIZE);
   Int_t lmts = 1;
   glLightModeliv(GL_LIGHT_MODEL_TWO_SIDE, &lmts);

   const Int_t  NP = fC->fNDiv;
   Int_t prev = NP - 1;
   Int_t i    = 0;
   Int_t next = 1;

   TEveVector curr_normal;
   TEveVector prev_normal;
   TMath::Cross((fP[next] - fP[prev]).Arr(), (fP[i] - fC->fApex).Arr(), prev_normal.Arr());

   prev = i; i = next; ++next;

   glBegin(GL_TRIANGLES);
   do
   {
      TMath::Cross((fP[next] - fP[prev]).Arr(), (fP[i] - fC->fApex).Arr(), curr_normal.Arr());

      glNormal3fv(prev_normal);
      glVertex3fv(fP[prev]);

      glNormal3fv(prev_normal + curr_normal);
      glVertex3fv(fC->fApex);

      glNormal3fv(curr_normal);
      glVertex3fv(fP[i]);

      prev_normal = curr_normal;

      prev = i;
      i    = next;
      ++next; if (next >= NP) next = 0;
   } while (prev != 0);
   glEnd();

   glPopAttrib();
}


//==============================================================================
// TEveJetConeProjectedGL
//==============================================================================

//______________________________________________________________________________
// OpenGL renderer class for TEveJetConeProjected.
//

ClassImp(TEveJetConeProjectedGL);

//______________________________________________________________________________
TEveJetConeProjectedGL::TEveJetConeProjectedGL() :
   TEveJetConeGL(), fM(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveJetConeProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveJetConeProjected>(obj);
   fC = dynamic_cast<TEveJetCone*>(fM->GetProjectable());
   return fC != 0;
}

//______________________________________________________________________________
void TEveJetConeProjectedGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveJetConeProjected*)fExternalObj)->AssertBBox());
}

namespace
{
   struct less_eve_vec_phi_t
   {
      bool operator()(const TEveVector& a, const TEveVector& b)
      { return a.Phi() < b.Phi(); }
   };
}

//______________________________________________________________________________
void TEveJetConeProjectedGL::CalculatePoints() const
{
   // Calculate points for drawing.

   static const TEveException kEH("TEveJetConeProjectedGL::CalculatePoints ");

   fP.resize(3);

   TEveProjection *proj = fM->GetManager()->GetProjection();

   switch (proj->GetType())
   {
      case TEveProjection::kPT_RPhi:
      {
         fP[0] = fC->fApex;
         fP[1] = fC->CalcBaseVec(TMath::Pi() + TMath::PiOver2());
         fP[2] = fC->CalcBaseVec(TMath::PiOver2());

         for (Int_t i = 0; i < 3; ++i)
            proj->ProjectVector(fP[i], fM->fDepth);

         break;
      }

      case TEveProjection::kPT_RhoZ:
      {
         fP[0] = fC->fApex;
         fP[1] = fC->CalcBaseVec(0);
         fP[2] = fC->CalcBaseVec(TMath::Pi());

         Float_t tm = fP[1].Theta();
         Float_t tM = fP[2].Theta();

         if (tM > fC->fThetaC && tm < fC->fThetaC)
         {
            fP.reserve(fP.size() + 1);
            TEveVector v(0, fC->fLimits.fY, fC->fLimits.fZ);
            fP.push_back(fC->CalcBaseVec(v.Eta(), fC->fPhi));
         }

         if (tM > TMath::Pi() - fC->fThetaC && tm < TMath::Pi() - fC->fThetaC)
         {
            fP.reserve(fP.size() + 1);
            TEveVector v(0, fC->fLimits.fY, -fC->fLimits.fZ);
            fP.push_back(fC->CalcBaseVec(v.Eta(), fC->fPhi));
         }

         const Int_t NP = fP.size();
         for (Int_t i = 0; i < NP; ++i)
            proj->ProjectVector(fP[i], fM->fDepth);

         std::sort(fP.begin() + 1, fP.end(), less_eve_vec_phi_t());

         break;
      }

      default:
         throw kEH + "Unsupported projection type.";
   }

}

//______________________________________________________________________________
void TEveJetConeProjectedGL::RenderOutline() const
{
   // Draw jet outline.

   const Int_t NP = fP.size();
   glBegin(GL_LINE_LOOP);
   for (Int_t i = 0; i < NP; ++i)
   {
      glVertex3fv(fP[i].Arr());
   }
   glEnd();
}

//______________________________________________________________________________
void TEveJetConeProjectedGL::RenderPolygon() const
{
   // Draw jet surface.

   const Int_t NP = fP.size();
   glBegin(GL_POLYGON);
   for (Int_t i = 0; i < NP; ++i)
   {
      glVertex3fv(fP[i].Arr());
   }
   glEnd();
}

//______________________________________________________________________________
void TEveJetConeProjectedGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw the cone.

   if (fP.empty()) CalculatePoints();

   if (rnrCtx.IsDrawPassOutlineLine())
   {
      RenderOutline();
   }
   else if (fM->fHighlightFrame && rnrCtx.Highlight())
   {
      if (fM->fDrawFrame)
      {
         TGLUtil::LineWidth(fM->fLineWidth);
         TGLUtil::Color(fM->fLineColor);
      }
      RenderOutline();
   }
   else
   {
      TGLObject::Draw(rnrCtx);
   }
}


//______________________________________________________________________________
void TEveJetConeProjectedGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Render with OpenGL.

   // printf("TEveJetConeProjectedGL::DirectDraw LOD %d\n", rnrCtx.CombiLOD());

   fMultiColor = (fM->fDrawFrame && fM->fFillColor != fM->fLineColor);

   glPushAttrib(GL_ENABLE_BIT);
   glDisable(GL_LIGHTING);

   if (fM->fDrawFrame)
   {
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.0f, 1.0f);
   }

   RenderPolygon();

   if (fM->fDrawFrame)
   {
      glEnable(GL_LINE_SMOOTH);

      TGLUtil::Color(fM->fLineColor);
      TGLUtil::LineWidth(fM->fLineWidth);
      RenderOutline();
   }

   glPopAttrib();
}
