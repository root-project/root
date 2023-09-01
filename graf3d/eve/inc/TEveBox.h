// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2010

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveBox
#define ROOT_TEveBox

#include "TEveShape.h"

//------------------------------------------------------------------------------
// TEveBox
//------------------------------------------------------------------------------

class TEveBox : public TEveShape
{
   friend class TEveBoxGL;

private:
   TEveBox(const TEveBox&);            // Not implemented
   TEveBox& operator=(const TEveBox&); // Not implemented

protected:
   Float_t fVertices[8][3];

public:
   TEveBox(const char* n="TEveBox", const char* t="");
   ~TEveBox() override;

   void SetVertex(Int_t i, Float_t x, Float_t y, Float_t z);
   void SetVertex(Int_t i, const Float_t* v);
   void SetVertices(const Float_t* vs);

   const Float_t* GetVertex(Int_t i) const { return fVertices[i]; }

   // For TAttBBox:
   void ComputeBBox() override;

   // Projectable:
   TClass* ProjectedClass(const TEveProjection* p) const override;

   ClassDefOverride(TEveBox, 0); // 3D box with arbitrary vertices.
};


//------------------------------------------------------------------------------
// TEveBoxProjected
//------------------------------------------------------------------------------

class TEveBoxProjected : public TEveShape,
                         public TEveProjected
{
   friend class TEveBoxProjectedGL;

private:
   TEveBoxProjected(const TEveBoxProjected&);            // Not implemented
   TEveBoxProjected& operator=(const TEveBoxProjected&); // Not implemented

protected:
   vVector2_t   fPoints;
   Int_t        fBreakIdx;
   vVector2_t   fDebugPoints;

   void SetDepthLocal(Float_t d) override;

   static Bool_t fgDebugCornerPoints;

public:
   TEveBoxProjected(const char* n="TEveBoxProjected", const char* t="");
   ~TEveBoxProjected() override;

   // For TAttBBox:
   void ComputeBBox() override;

   // Projected:
   void SetProjection(TEveProjectionManager* mng, TEveProjectable* model) override;
   void UpdateProjection() override;

   TEveElement* GetProjectedAsElement() override { return this; }

   static Bool_t GetDebugCornerPoints();
   static void   SetDebugCornerPoints(Bool_t d);

   ClassDefOverride(TEveBoxProjected, 0); // Projection of TEveBox.
};

#endif
