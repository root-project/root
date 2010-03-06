// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveShape.h"
#include "Riostream.h"

//______________________________________________________________________________
// Description of TEveShape
//

ClassImp(TEveShape);

//______________________________________________________________________________
TEveShape::TEveShape(const char* n, const char* t) :
   TEveElementList(n, t),
   fFillColor(5),
   fLineColor(3),
   fLineWidth(1),
   fDrawFrame(kTRUE),
   fHighlightFrame(kTRUE)
{
   // Constructor.

   SetMainColorPtr(&fFillColor);
}

//______________________________________________________________________________
TEveShape::~TEveShape()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveShape::SetMainColor(Color_t color)
{
   // Set main color.
   // Override so that line-color can also be changed if it is equal
   // to fill color (which is treated as main color).

   if (fFillColor == fLineColor) {
      fLineColor = color;
      StampObjProps();
   }
   TEveElementList::SetMainColor(color);
}

//______________________________________________________________________________
void TEveShape::CopyVizParams(const TEveElement* el)
{
   // Copy visualization parameters from element el.

   const TEveShape* m = dynamic_cast<const TEveShape*>(el);
   if (m)
   {
      fFillColor = m->fFillColor;
      fLineColor = m->fLineColor;
      fLineWidth = m->fLineWidth;
      fHighlightFrame = m->fHighlightFrame;
   }

   TEveElementList::CopyVizParams(el);
}

//______________________________________________________________________________
void TEveShape::WriteVizParams(ostream& out, const TString& var)
{
   // Write visualization parameters.

   TEveElementList::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   out << t << "SetFillColor(" << fFillColor << ");\n";
   out << t << "SetLineColor(" << fLineColor << ");\n";
   out << t << "SetLineWidth(" << fLineWidth << ");\n";
   out << t << "SetHighlightFrame(" << ToString(fHighlightFrame) << ");\n";
}

//______________________________________________________________________________
void TEveShape::Paint(Option_t*)
{
   // Paint this object. Only direct rendering is supported.

   PaintStandard(this);
}
