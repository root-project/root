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

#include "TEveElement.h"
#include "TEveVSDStructs.h"
#include "TAttBBox.h"

class TEveJetCone : public TEveElementList,
                    public TAttBBox
{
   friend class TEveJetConeGL;

private:
   TEveJetCone(const TEveJetCone&);            // Not implemented
   TEveJetCone& operator=(const TEveJetCone&); // Not implemented

   void    FillTEveVectorFromEtaPhi( TEveVector &vec, const Float_t& eta, const Float_t& phi );
   Float_t GetArcCosConeOpeningAngle( const TEveVector& axis, const TEveVector& contour );

protected:
   typedef std::vector<TEveVector>        vTEveVector_t;
   typedef vTEveVector_t::iterator        vTEveVector_i;
   typedef vTEveVector_t::const_iterator  vTEveVector_ci;

   TEveVector      fApex;             // Apex of the cone, initialized to ( 0., 0., 0. )
   vTEveVector_t   fBasePoints;       // List of contour points
   TEveVector      fCylinderBorder;   // Border of Barrel/Cylinder to cut the cone
   Float_t         fThetaC;           // Angle between axis and  the edge of top-side of cylinder

public:
   TEveJetCone(const Text_t* n="TEveJetCone", const Text_t* t="");
   virtual ~TEveJetCone() {}

   void SetApex(const TEveVector& a)                      { fApex = a; }  // Sets apex of cone
   void SetCylinder( const Float_t& r, const Float_t& z ) {
      fCylinderBorder.Set( r, 0.f, z ); fThetaC = fCylinderBorder.Theta(); } // Set border cylinder

   Int_t AddCone(Float_t eta, Float_t phi, Float_t coneRadius, Float_t height=-1);
   Int_t AddEllipticCone(Float_t eta, Float_t phi, Float_t reta, Float_t rphi, Float_t height=-1);

   virtual Bool_t  CanEditMainTransparency() const { return kTRUE; }

   // For TAttBBox:
   virtual void ComputeBBox();
   // If painting is needed:
   virtual void Paint(Option_t* option="");

   ClassDef(TEveJetCone, 0); // Short description.
};

#endif
