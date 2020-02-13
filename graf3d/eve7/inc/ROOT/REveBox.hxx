// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2010

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveBox
#define ROOT_REveBox

#include "ROOT/REveShape.hxx"

namespace ROOT {
namespace Experimental {

//------------------------------------------------------------------------------
// REveBox
//------------------------------------------------------------------------------

class REveBox : public REveShape
{
private:
   REveBox(const REveBox&);            // Not implemented
   REveBox& operator=(const REveBox&); // Not implemented

protected:
   Float_t fVertices[8][3];

public:
   REveBox(const char* n="REveBox", const char* t="");
   virtual ~REveBox();

   void SetVertex(Int_t i, Float_t x, Float_t y, Float_t z);
   void SetVertex(Int_t i, const Float_t* v);
   void SetVertices(const Float_t* vs);

   const Float_t* GetVertex(Int_t i) const { return fVertices[i]; }

   // For TAttBBox:
   virtual void ComputeBBox();

   // Projectable:
   virtual TClass* ProjectedClass(const REveProjection* p) const;

   ClassDef(REveBox, 0); // 3D box with arbitrary vertices.
};


//------------------------------------------------------------------------------
// REveBoxProjected
//------------------------------------------------------------------------------

class REveBoxProjected : public REveShape,
                         public REveProjected
{
private:
   REveBoxProjected(const REveBoxProjected&);            // Not implemented
   REveBoxProjected& operator=(const REveBoxProjected&); // Not implemented

protected:
   vVector2_t   fPoints;
   Int_t        fBreakIdx;
   vVector2_t   fDebugPoints;

   virtual void SetDepthLocal(Float_t d);

   static Bool_t fgDebugCornerPoints;

public:
   REveBoxProjected(const char* n="REveBoxProjected", const char* t="");
   virtual ~REveBoxProjected();

   // For TAttBBox:
   virtual void ComputeBBox();

   // Projected:
   virtual void SetProjection(REveProjectionManager* mng, REveProjectable* model);
   virtual void UpdateProjection();

   virtual REveElement* GetProjectedAsElement() { return this; }

   static Bool_t GetDebugCornerPoints();
   static void   SetDebugCornerPoints(Bool_t d);

   ClassDef(REveBoxProjected, 0); // Projection of REveBox.
};

} // namespace Experimental
} // namespace ROOT
#endif
