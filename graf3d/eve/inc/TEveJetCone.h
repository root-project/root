// @(#)root/eve:$Id$
// Author: Matevz Tadel, Jochen Thaeder 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveJetCone
#define ROOT_TEveJetCone

#include "TEveShape.h"
#include "TEveVector.h"


//------------------------------------------------------------------------------
// TEveJetCone
//------------------------------------------------------------------------------

class TEveJetCone : public TEveShape
{
   friend class TEveJetConeProjected;
   friend class TEveJetConeGL;
   friend class TEveJetConeProjectedGL;

private:
   TEveJetCone(const TEveJetCone&);            // Not implemented
   TEveJetCone& operator=(const TEveJetCone&); // Not implemented

protected:
   TEveVector      fApex;        // Apex of the cone.
   TEveVector      fAxis;        // Axis of the cone.
   TEveVector      fLimits;      // Border of Barrel/Cylinder to cut the cone.
   Float_t         fThetaC;      // Transition theta
   Float_t         fEta,  fPhi;
   Float_t         fDEta, fDPhi;
   Int_t           fNDiv;

   TEveVector CalcEtaPhiVec(Float_t eta, Float_t phi) const;
   TEveVector CalcBaseVec  (Float_t eta, Float_t phi) const;
   TEveVector CalcBaseVec  (Float_t alpha) const;
   Bool_t     IsInTransitionRegion() const;

public:
   TEveJetCone(const Text_t* n="TEveJetCone", const Text_t* t="");
   virtual ~TEveJetCone() {}

   virtual void    ComputeBBox();
   virtual TClass* ProjectedClass(const TEveProjection* p) const;

   void  SetApex(const TEveVector& a)      { fApex = a; }
   void  SetCylinder(Float_t r, Float_t z) { fLimits.Set(0, r, z); fThetaC = fLimits.Theta(); }
   void  SetRadius  (Float_t r)            { fLimits.Set(r, 0, 0); fThetaC = 10; }

   Int_t GetNDiv() const  { return fNDiv; }
   void  SetNDiv(Int_t n) { fNDiv = TMath::Max(3, n); }

   Int_t AddCone(Float_t eta, Float_t phi, Float_t cone_r, Float_t length=0);
   Int_t AddEllipticCone(Float_t eta, Float_t phi, Float_t reta, Float_t rphi, Float_t length=0);

   ClassDef(TEveJetCone, 0); // Short description.
};


//------------------------------------------------------------------------------
// TEveJetConeProjected
//------------------------------------------------------------------------------

class TEveJetConeProjected : public TEveShape,
                             public TEveProjected
{
   friend class TEveJetConeProjectedGL;

private:
   TEveJetConeProjected(const TEveJetConeProjected&);            // Not implemented
   TEveJetConeProjected& operator=(const TEveJetConeProjected&); // Not implemented

protected:
   virtual void SetDepthLocal(Float_t d);

public:
   TEveJetConeProjected(const char* n="TEveJetConeProjected", const char* t="");
   virtual ~TEveJetConeProjected();

   // For TAttBBox:
   virtual void ComputeBBox();

   // Projected:
   virtual void SetProjection(TEveProjectionManager* mng, TEveProjectable* model);
   virtual void UpdateProjection();

   virtual TEveElement* GetProjectedAsElement() { return this; }

   ClassDef(TEveJetConeProjected, 0); // Projection of TEveJetCone.
};

#endif
