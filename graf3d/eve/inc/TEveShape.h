// @(#)root/eve:$Id$
// Author: Matevz Tadel 2010

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveShape
#define ROOT_TEveShape

#include "TEveElement.h"

#include "TAtt3D.h"
#include "TAttBBox.h"
#include "TColor.h"

class TEveShape : public TEveElementList,
                  public TAtt3D,
                  public TAttBBox
{
private:
   TEveShape(const TEveShape&);            // Not implemented
   TEveShape& operator=(const TEveShape&); // Not implemented

protected:
   Color_t      fFillColor; // fill color of polygons
   Color_t      fLineColor; // outline color of polygons
   Float_t      fLineWidth; // outline width of polygons

   Bool_t       fHighlightFrame; // higlight mode

public:
   TEveShape(const char* n="TEveShape", const char* t="");
   virtual ~TEveShape();

   // Rendering parameters.
   virtual Bool_t  CanEditMainColor() const { return kTRUE; }
   virtual void    SetMainColor(Color_t color);

   virtual Bool_t  CanEditMainTransparency() const { return kTRUE; }

   virtual Color_t GetFillColor() const { return fFillColor; }
   virtual Color_t GetLineColor() const { return fLineColor; }
   virtual Float_t GetLineWidth() const { return fLineWidth;}
   virtual Bool_t  GetHighlightFrame() const { return fHighlightFrame; }

   virtual void    SetFillColor(Color_t c)  { fFillColor = c; }
   virtual void    SetLineColor(Color_t c)  { fLineColor = c; }
   virtual void    SetLineWidth(Float_t lw) { fLineWidth = lw;}
   virtual void    SetHighlightFrame(Bool_t f) { fHighlightFrame = f; }

   // Abstract function from TAttBBox:
   // virtual void ComputeBBox();

   // Virtual from TObject
   virtual void Paint(Option_t* option="");

   ClassDef(TEveShape, 0); // Abstract base-class for 2D/3D shapes.
};

#endif
