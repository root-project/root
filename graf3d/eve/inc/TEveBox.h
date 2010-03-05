// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

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

   // For TAttBBox:
   virtual void ComputeBBox();

   ClassDef(TEveBox, 0); // Short description.
};

#endif
