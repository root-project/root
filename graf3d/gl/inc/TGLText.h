// @(#)root/gl:$Id$
// Author:  Olivier Couet 12/04/2007

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLText
#define ROOT_TGLText

#include "TAttText.h"

class FTFont;

class TGLText : public TAttText {
private:
   TGLText(const TGLText&);            // Not implemented
   TGLText& operator=(const TGLText&); // Not implemented

   Double_t fX; // X position
   Double_t fY; // Y position
   Double_t fZ; // Z position
   Double_t fAngle1; // 1st angle.
   Double_t fAngle2; // 2nd angle.
   Double_t fAngle3; // 3rd angle.
   FTFont* fGLTextFont;

public:
   TGLText();
   TGLText(Double_t x, Double_t y, Double_t z, const char *text);
   virtual ~TGLText();

   FTFont* GetFont() { return fGLTextFont; }

   void SetGLTextAngles(Double_t a1, Double_t a2, Double_t a3);
   void SetGLTextFont(Font_t fontnumber);
   void PaintGLText(Double_t x, Double_t y, Double_t z, const char *text);
   void PaintBBox(const char *text);
   void BBox(const char* string, float& llx, float& lly, float& llz,
                                 float& urx, float& ury, float& urz);

   ClassDef(TGLText,0) // a GL text
};

#endif
