// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   26/01/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QuartzText
#define ROOT_QuartzText

#include <vector>

#include <Cocoa/Cocoa.h>

#include "CocoaUtils.h"
#include "GuiTypes.h"


/////////////////////////////////////////////////
//                                             //
// TextLine - wrapper class for a CoreText's   //
// CTLine: hide all the API, calls, objects,   //
// CoreFoundation objects required to draw     //
// a simple line of text and be able to        //
// calculate text metrics.                     //
//                                             //
/////////////////////////////////////////////////

namespace ROOT {
namespace Quartz {

// Core Text's CTLine wrapper.
class TextLine {
public:
   TextLine(const char *textLine, CTFontRef font);
   TextLine(const std::vector<UniChar> &textLine, CTFontRef font);

   TextLine(const char *textLine, CTFontRef font, Color_t color);
   TextLine(const std::vector<UniChar> &textLine, CTFontRef font, Color_t color);
   TextLine(const char *textLine, CTFontRef font, const CGFloat *rgb);

   ~TextLine();

   void GetBounds(UInt_t &w, UInt_t &h)const;
   void GetAscentDescent(Int_t &asc, Int_t &desc)const;

   void DrawLine(CGContextRef ctx)const;
   void DrawLine(CGContextRef ctx, Double_t x, Double_t y)const;
private:
   CTLineRef fCTLine; //Core Text line, created from Attributed string.
   CTFontRef fCTFont; //A font used for this CTLine.

   void Init(const char *textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values);
   void Init(const std::vector<UniChar> &textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values);

   TextLine(const TextLine &rhs);
   TextLine &operator = (const TextLine &rhs);
};

//Aux. function which extracts glyphs, calculates their positions, draws glyphs with manual layout (GUI text).
void DrawTextLineNoKerning(CGContextRef ctx, CTFontRef font, const std::vector<UniChar> &text, Int_t x, Int_t y);

}
}

#endif
