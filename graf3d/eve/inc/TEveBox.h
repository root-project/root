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
   virtual ~TEveBox();

   void SetVertex(Int_t i, Float_t x, Float_t y, Float_t z);
   void SetVertex(Int_t i, const Float_t* v);
   void SetVertices(const Float_t* vs);

   const Float_t* GetVertex(Int_t i) const { return fVertices[i]; }

   // For TAttBBox:
   virtual void ComputeBBox();

   // Projectable:
   virtual TClass* ProjectedClass(const TEveProjection* p) const;

   ClassDef(TEveBox, 0); // 3D box with arbitrary vertices.
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

   virtual void SetDepthLocal(Float_t d);

   static Bool_t fgDebugCornerPoints;

public:
   TEveBoxProjected(const char* n="TEveBoxProjected", const char* t="");
   virtual ~TEveBoxProjected();

   // For TAttBBox:
   virtual void ComputeBBox();

   // Projected:
   virtual void SetProjection(TEveProjectionManager* mng, TEveProjectable* model);
   virtual void UpdateProjection();

   virtual TEveElement* GetProjectedAsElement() { return this; }

   static Bool_t GetDebugCornerPoints();
   static void   SetDebugCornerPoints(Bool_t d);

   ClassDef(TEveBoxProjected, 0); // Projection of TEveBox.
};

#endif
