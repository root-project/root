// @(#)root/eve:$Id$
// Author: Matevz Tadel, Jochen Thaeder 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TEveJetCone.hxx"
#include "ROOT/TEveTrans.hxx"
#include "ROOT/TEveProjectionManager.hxx"

#include "TMath.h"

#include <cassert>

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class TEveJetCone
\ingroup TEve
Draws a jet cone with leading particle is specified in (eta,phi) and
cone radius is given.

If Apex is not set, default is (0.,0.,0.)
In case of cylinder was set, cone is cut at the cylinder edges.

Example :
~~~ {.cpp}
  Float_t coneEta    = r.Uniform(-0.9, 0.9);
  Float_t conePhi    = r.Uniform(0.0, TwoPi() );
  Float_t coneRadius = 0.4;

  TEveJetCone* jetCone = new TEveJetCone("JetCone");
  jetCone->SetCylinder(250, 250);
  if (jetCone->AddCone(coneEta, conePhi, coneRadius) != -1)
    gEve->AddElement(jetCone);
~~~

#### Implementation notes

TEveVector fLimits encodes the following information:
  - fY, fZ:  barrel radius and endcap z-position;
             if both are 0, fX encodes the spherical radius
  - fX    :  scaling for length of the cone
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveJetCone::TEveJetCone(const Text_t* n, const Text_t* t) :
   TEveShape(n, t),
   fApex(),
   fLimits(), fThetaC(10),
   fEta(0), fPhi(0), fDEta(0), fDPhi(0), fNDiv(72)
{
   fColor = kGreen;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t TEveJetCone::WriteCoreJson(nlohmann::json& j, Int_t rnr_offset)
{
   Int_t ret = TEveElement::WriteCoreJson(j, rnr_offset);

   j["fMainColor"] = GetFillColor();
   j["fLineColor"] = GetLineColor();

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Crates 3D point array for rendering.

void TEveJetCone::BuildRenderData()
{
   assert(fNDiv > 2);

   const Int_t  NP = 1 + fNDiv;

   RenderData *rd = new RenderData("makeJet", 3 * NP);

   rd->PushV(fApex);

   Float_t angle_step = TMath::TwoPi() / fNDiv;
   Float_t angle      = 0;
   for (Int_t i = 0; i < fNDiv; ++i, angle += angle_step)
   {
      rd->PushV( CalcBaseVec(angle) );
   }

   fRenderData.reset(rd);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding-box of the data.

void TEveJetCone::ComputeBBox()
{
   BBoxInit();
   BBoxCheckPoint(fApex);
   BBoxCheckPoint(CalcBaseVec(0));
   BBoxCheckPoint(CalcBaseVec(TMath::PiOver2()));
   BBoxCheckPoint(CalcBaseVec(TMath::Pi()));
   BBoxCheckPoint(CalcBaseVec(TMath::Pi() + TMath::PiOver2()));
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveProjectable, returns TEveJetConeProjected class.

TClass* TEveJetCone::ProjectedClass(const TEveProjection*) const
{
   return TEveJetConeProjected::Class();
}

////////////////////////////////////////////////////////////////////////////////
/// Add jet cone.
/// parameters are :
/// - (eta,phi)    : of the center/leading particle
/// - cone_r       : cone radius in eta-phi space
/// - length       : length of the cone
///   - if cylinder is set and length is adapted to cylinder.
///      - if length is given, it will be used as scalar factor
///   - if cylinder is not set, length is used as length of the cone
/// Return 0 on success.

Int_t TEveJetCone::AddCone(Float_t eta, Float_t phi, Float_t cone_r, Float_t length)
{
   return AddEllipticCone(eta, phi, cone_r, cone_r, length);
}

////////////////////////////////////////////////////////////////////////////////
/// Add jet cone.
/// parameters are :
/// - (eta,phi)    : of the center/leading particle
/// - (reta, rphi) : radius of cone in eta-phi space
/// - length       : length of the cone
///   - if cylinder is set and length is adapted to cylinder.
///      - if length is given, it will be used as scalar factor
///   - if cylinder is not set, length is used as length of the cone
/// Returns 0 on success.

Int_t TEveJetCone::AddEllipticCone(Float_t eta, Float_t phi, Float_t reta, Float_t rphi, Float_t length)
{
   if (length != 0) fLimits.fX = length;

   if (fLimits.IsZero())
      return -1;

   fEta = eta; fPhi = phi; fDEta = reta; fDPhi = rphi;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill TEveVector with eta and phi, magnitude 1.

TEveVector TEveJetCone::CalcEtaPhiVec(Float_t eta, Float_t phi) const
{
   using namespace TMath;

   return TEveVector(Cos(phi) / CosH(eta), Sin(phi) / CosH(eta), TanH(eta));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns point on the base of the cone with given eta and phi.

TEveVector TEveJetCone::CalcBaseVec(Float_t eta, Float_t phi) const
{
   using namespace TMath;

   TEveVector vec = CalcEtaPhiVec(eta, phi);

   // -- Set length of the contourPoint
   if (fLimits.fY != 0 && fLimits.fZ != 0)
   {
      Float_t theta = vec.Theta();
      if (theta < fThetaC)
         vec *= fLimits.fZ / Cos(theta);
      else if (theta > Pi() - fThetaC)
         vec *= fLimits.fZ / Cos(theta - Pi());
      else
         vec *= fLimits.fY / Sin(theta);

      if (fLimits.fX != 0) vec *= fLimits.fX;
   }
   else
   {
      vec *= fLimits.fX;
   }

   return vec;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns point on the base of the cone with internal angle alpha:
/// alpha = 0 -> max eta,  alpha = pi/2 -> max phi, ...

TEveVector TEveJetCone::CalcBaseVec(Float_t alpha) const
{
   using namespace TMath;

   return CalcBaseVec(fEta + fDEta * Cos(alpha), fPhi + fDPhi * Sin(alpha));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if the cone is in barrel / endcap transition region.

Bool_t TEveJetCone::IsInTransitionRegion() const
{
   using namespace TMath;

   Float_t tm = CalcBaseVec(0).Theta();
   Float_t tM = CalcBaseVec(Pi()).Theta();

   return (tM > fThetaC        && tm < fThetaC) ||
          (tM > Pi() - fThetaC && tm < Pi() - fThetaC);
}


/** \class TEveJetConeProjected
\ingroup TEve
Projection of TEveJetCone.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveJetConeProjected::TEveJetConeProjected(const char* n, const char* t) :
   TEveShape(n, t)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveJetConeProjected::~TEveJetConeProjected()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t TEveJetConeProjected::WriteCoreJson(nlohmann::json& j, Int_t rnr_offset)
{
   Int_t ret = TEveElement::WriteCoreJson(j, rnr_offset);

   j["fMainColor"] = GetFillColor();
   j["fLineColor"] = GetLineColor();

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Crates 3D point array for rendering.

void TEveJetConeProjected::BuildRenderData()
{
   static const TEveException kEH("TEveJetConeProjected::BuildRenderData ");

   TEveProjection *P = GetManager()->GetProjection();
   TEveJetCone    *C = dynamic_cast<TEveJetCone*>(GetProjectable());

   RenderData *rd = new RenderData("makeJetProjected", 4);

   std::vector<TEveVector> V; V.reserve(4); V.resize(3);

   switch (P->GetType())
   {
      case TEveProjection::kPT_RPhi:
      {
         V[0] = C->fApex;
         V[1] = C->CalcBaseVec(TMath::Pi() + TMath::PiOver2());
         V[2] = C->CalcBaseVec(TMath::PiOver2());

         for (Int_t i = 0; i < 3; ++i)
            P->ProjectVector(V[i], fDepth);

         break;
      }

      case TEveProjection::kPT_RhoZ:
      {
         V[0] = C->fApex;
         V[1] = C->CalcBaseVec(0);
         V[2] = C->CalcBaseVec(TMath::Pi());

         Float_t tm = V[1].Theta();
         Float_t tM = V[2].Theta();

         if (tM > C->fThetaC && tm < C->fThetaC)
         {
            TEveVector v(0, C->fLimits.fY, C->fLimits.fZ);

            V.push_back(C->CalcBaseVec(v.Eta(), C->fPhi));
         }

         if (tM > TMath::Pi() - C->fThetaC && tm < TMath::Pi() - C->fThetaC)
         {
            TEveVector v(0, C->fLimits.fY, -C->fLimits.fZ);

            V.push_back(C->CalcBaseVec(v.Eta(), C->fPhi));
         }

         for (auto &v : V) P->ProjectVector(v, fDepth);

         std::sort(V.begin() + 1, V.end(),
                   [](const auto& a, const auto &b) -> bool
                      { return a.Phi() < b.Phi(); });

         break;
      }

      default:
         throw kEH + "Unsupported projection type.";
   }

   for (auto &v : V) rd->PushV(v);

   fRenderData.reset(rd);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding-box, virtual from TAttBBox.

void TEveJetConeProjected::ComputeBBox()
{
   BBoxInit();

   TEveJetCone    *cone = dynamic_cast<TEveJetCone*>(fProjectable);
   TEveProjection *proj = GetManager()->GetProjection();
   TEveVector v;
   v = cone->fApex;                                       proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(0);                              proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(TMath::PiOver2());               proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(TMath::Pi());                    proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(TMath::Pi() + TMath::PiOver2()); proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
}

////////////////////////////////////////////////////////////////////////////////
/// This is virtual method from base-class TEveProjected.

void TEveJetConeProjected::SetDepthLocal(Float_t d)
{
   SetDepthCommon(d, this, fBBox);
}

////////////////////////////////////////////////////////////////////////////////
/// This is virtual method from base-class TEveProjected.

void TEveJetConeProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   TEveProjected::SetProjection(mng, model);
   CopyVizParams(dynamic_cast<TEveElement*>(model));
}

////////////////////////////////////////////////////////////////////////////////
/// Re-project the jet-cone.

void TEveJetConeProjected::UpdateProjection()
{
}
