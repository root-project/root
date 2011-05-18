// @(#)root/eve:$Id$
// Author: Matevz Tadel, Jochen Thaeder 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveJetCone.h"
#include "TEveTrans.h"
#include "TEveProjectionManager.h"

#include "TMath.h"


//==============================================================================
// TEveJetCone
//==============================================================================

//______________________________________________________________________________
//
// Draws a jet cone with leading particle is specified in (eta,phi) and
// cone radius is given.
//
// If Apex is not set, default is (0.,0.,0.)
// In case of cylinder was set, cone is cut at the cylinder edges.
//
// Example :
//
//  Float_t coneEta    = r.Uniform(-0.9, 0.9);
//  Float_t conePhi    = r.Uniform(0.0, TwoPi() );
//  Float_t coneRadius = 0.4;
//
//  TEveJetCone* jetCone = new TEveJetCone("JetCone");
//  jetCone->SetCylinder(250, 250);
//  if (jetCone->AddCone(coneEta, conePhi, coneRadius) != -1)
//    gEve->AddElement(jetCone);
//
//
// Implementation notes
//
// TEveVector fLimits encodes the following information:
//   fY, fZ:  barrel radius and endcap z-position;
//            if both are 0, fX encodes the spherical radius
//   fX    :  scaling for length of the cone

ClassImp(TEveJetCone);

//______________________________________________________________________________
TEveJetCone::TEveJetCone(const Text_t* n, const Text_t* t) :
   TEveShape(n, t),
   fApex(),
   fLimits(), fThetaC(10),
   fEta(0), fPhi(0), fDEta(0), fDPhi(0), fNDiv(72)
{
   // Constructor.

   fColor = kGreen;
}

//______________________________________________________________________________
void TEveJetCone::ComputeBBox()
{
   // Compute bounding-box of the data.

   BBoxInit();
   BBoxCheckPoint(fApex);
   BBoxCheckPoint(CalcBaseVec(0));
   BBoxCheckPoint(CalcBaseVec(TMath::PiOver2()));
   BBoxCheckPoint(CalcBaseVec(TMath::Pi()));
   BBoxCheckPoint(CalcBaseVec(TMath::Pi() + TMath::PiOver2()));
}

//______________________________________________________________________________
TClass* TEveJetCone::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, returns TEveJetConeProjected class.

   return TEveJetConeProjected::Class();
}


//______________________________________________________________________________
Int_t TEveJetCone::AddCone(Float_t eta, Float_t phi, Float_t cone_r, Float_t length)
{
   // Add jet cone.
   // parameters are :
   // * (eta,phi)    : of the center/leading particle
   // * cone_r       : cone radius in eta-phi space
   // * length       : length of the cone
   //   * if cylinder is set and length is adapted to cylinder.
   //     - if length is given, it will be used as scalar factor
   //   * if cylinder is not set, length is used as length of the cone
   // Return 0 on sucess.

   return AddEllipticCone(eta, phi, cone_r, cone_r, length);
}

//______________________________________________________________________________
Int_t TEveJetCone::AddEllipticCone(Float_t eta, Float_t phi, Float_t reta, Float_t rphi, Float_t length)
{
   // Add jet cone.
   // parameters are :
   // * (eta,phi)    : of the center/leading particle
   // * (reta, rphi) : radius of cone in eta-phi space
   // * length       : length of the cone
   //   * if cylinder is set and length is adapted to cylinder.
   //     - if length is given, it will be used as scalar factor
   //   * if cylinder is not set, length is used as length of the cone
   // Returns 0 on sucess.

   using namespace TMath;

   if (length != 0) fLimits.fX = length;

   if (fLimits.IsZero())
      return -1;

   fEta = eta; fPhi = phi; fDEta = reta; fDPhi = rphi;

   return 0;
}

//______________________________________________________________________________
TEveVector TEveJetCone::CalcEtaPhiVec(Float_t eta, Float_t phi) const
{
   // Fill TEveVector with eta and phi, magnitude 1.

   using namespace TMath;

   return TEveVector(Cos(phi) / CosH(eta), Sin(phi) / CosH(eta), TanH(eta));
}

//______________________________________________________________________________
TEveVector TEveJetCone::CalcBaseVec(Float_t eta, Float_t phi) const
{
   // Returns point on the base of the cone with given eta and phi.

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

//______________________________________________________________________________
TEveVector TEveJetCone::CalcBaseVec(Float_t alpha) const
{
   // Returns point on the base of the cone with internal angle alpha:
   // alpha = 0 -> max eta,  alpha = pi/2 -> max phi, ...

   using namespace TMath;

   return CalcBaseVec(fEta + fDEta * Cos(alpha), fPhi + fDPhi * Sin(alpha));
}

//______________________________________________________________________________
Bool_t TEveJetCone::IsInTransitionRegion() const
{
   // Returns true if the cone is in barrel / endcap transition region.

   using namespace TMath;

   Float_t tm = CalcBaseVec(0).Theta();
   Float_t tM = CalcBaseVec(Pi()).Theta();

   return (tM > fThetaC        && tm < fThetaC) ||
          (tM > Pi() - fThetaC && tm < Pi() - fThetaC);
}

//==============================================================================
// TEveJetConeProjected
//==============================================================================

//______________________________________________________________________________
//
// Projection of TEveJetCone.

//______________________________________________________________________________
TEveJetConeProjected::TEveJetConeProjected(const char* n, const char* t) :
   TEveShape(n, t)
{
   // Constructor.
}

//______________________________________________________________________________
TEveJetConeProjected::~TEveJetConeProjected()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveJetConeProjected::ComputeBBox()
{
   // Compute bounding-box, virtual from TAttBBox.

   BBoxInit();

   TEveJetCone    *cone = dynamic_cast<TEveJetCone*>(fProjectable);
//______________________________________________________________________________
   TEveProjection *proj = GetManager()->GetProjection();
   TEveVector v;
   v = cone->fApex;                                       proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(0);                              proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(TMath::PiOver2());               proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(TMath::Pi());                    proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
   v = cone->CalcBaseVec(TMath::Pi() + TMath::PiOver2()); proj->ProjectVector(v, fDepth); BBoxCheckPoint(v);
}

//______________________________________________________________________________
void TEveJetConeProjected::SetDepthLocal(Float_t d)
{
   // This is virtual method from base-class TEveProjected.

   SetDepthCommon(d, this, fBBox);
}

//______________________________________________________________________________
void TEveJetConeProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   // This is virtual method from base-class TEveProjected.

   TEveProjected::SetProjection(mng, model);
   CopyVizParams(dynamic_cast<TEveElement*>(model));
}

//______________________________________________________________________________
void TEveJetConeProjected::UpdateProjection()
{
   // Re-project the jet-cone.

}
