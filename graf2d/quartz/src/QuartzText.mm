// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   26/01/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>
#include <cassert>
#include <vector>
#include <cmath>

#include <Availability.h>

#include "QuartzText.h"
#include "CocoaUtils.h"
#include "TVirtualX.h"
#include "TColor.h"
#include "TError.h"
#include "TROOT.h"
#include "TMath.h"

namespace ROOT {
namespace Quartz {

#ifdef MAC_OS_X_VERSION_10_11

const CTFontOrientation defaultFontOrientation = kCTFontOrientationDefault;
const CTFontOrientation horizontalFontOrientation = kCTFontOrientationHorizontal;
const CTFontOrientation verticalFontOrientation = kCTFontOrientationVertical;

#else
// Constants deprecated starting from 10.11
const CTFontOrientation defaultFontOrientation = kCTFontDefaultOrientation;
const CTFontOrientation horizontalFontOrientation = kCTFontHorizontalOrientation;
const CTFontOrientation verticalFontOrientation = kCTFontVerticalOrientation;

#endif

namespace {

//______________________________________________________________________________
void GetTextColorForIndex(Color_t colorIndex, Float_t &r, Float_t &g, Float_t &b, Float_t &a)
{
   if (const TColor * const color = gROOT->GetColor(colorIndex)) {
      color->GetRGB(r, g, b);
      a = color->GetAlpha();
   }
}

//_________________________________________________________________
CGRect BBoxForCTRun(CTFontRef font, CTRunRef run)
{
   assert(font != 0 && "BBoxForCTRun, parameter 'font' is null");
   assert(run != 0 && "BBoxForCTRun, parameter 'run' is null");

   CGRect bbox = {};
   if (const CFIndex nGlyphs = CTRunGetGlyphCount(run)) {
      std::vector<CGGlyph> glyphs(nGlyphs);
      CTRunGetGlyphs(run, CFRangeMake(0, 0), &glyphs[0]);
      bbox = CTFontGetBoundingRectsForGlyphs(font, defaultFontOrientation,
                                             &glyphs[0], 0, nGlyphs);
   }

   return bbox;
}

}

//_________________________________________________________________
TextLine::TextLine(const char *textLine, CTFontRef font)
             : fCTLine(0),
               fCTFont(font)
{
   //TODO: why don't I have asserts on parameters here?

   //Create attributed string with one attribue: the font.
   CFStringRef keys[] = {kCTFontAttributeName};
   CFTypeRef values[] = {font};

   Init(textLine, 1, keys, values);
}

//_________________________________________________________________
TextLine::TextLine(const std::vector<UniChar> &unichars, CTFontRef font)
             : fCTLine(0),
               fCTFont(font)
   
{
   //TODO: why don't I have asserts on parameters here?

   //Create attributed string with one attribue: the font.
   CFStringRef keys[] = {kCTFontAttributeName};
   CFTypeRef values[] = {font};

   Init(unichars, 1, keys, values);
}

//_________________________________________________________________
TextLine::TextLine(const char *textLine, CTFontRef font, Color_t color)
            : fCTLine(0),
              fCTFont(font)
{
   //TODO: why don't I have asserts on parameters here?

   //Create attributed string with font and color.
   using MacOSX::Util::CFScopeGuard;

   const CFScopeGuard<CGColorSpaceRef> rgbColorSpace(CGColorSpaceCreateDeviceRGB());
   if (!rgbColorSpace.Get())
      throw std::runtime_error("TextLine: color space");

   Float_t rgba[] = {0.f, 0.f, 0.f, 1.f};
   GetTextColorForIndex(color, rgba[0], rgba[1], rgba[2], rgba[3]);
   const CGFloat cgRgba[] = {rgba[0], rgba[1], rgba[2], rgba[3]};

   const CFScopeGuard<CGColorRef> textColor(CGColorCreate(rgbColorSpace.Get(), cgRgba));
   //Not clear from docs, if textColor.Get() can be 0.
   
   CFStringRef keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
   CFTypeRef values[] = {font, textColor.Get()};

   Init(textLine, 2, keys, values);
}

//_________________________________________________________________
TextLine::TextLine(const char *textLine, CTFontRef font, const CGFloat *rgb)
            : fCTLine(0),
              fCTFont(font)
{
   //TODO: why don't I have asserts on parameters here?

   //Create attributed string with font and color.
   using ROOT::MacOSX::Util::CFScopeGuard;
   CFScopeGuard<CGColorSpaceRef> rgbColorSpace(CGColorSpaceCreateDeviceRGB());
   
   if (!rgbColorSpace.Get())
      throw std::runtime_error("TexLine: color space is null");

   CFScopeGuard<CGColorRef> textColor(CGColorCreate(rgbColorSpace.Get(), rgb));
   //Not clear from docs, if textColor can be 0.

   CFStringRef keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
   CFTypeRef values[] = {font, textColor.Get()};

   Init(textLine, 2, keys, values);
}

//_________________________________________________________________
TextLine::TextLine(const std::vector<UniChar> &unichars, CTFontRef font, Color_t color)
            : fCTLine(0),
              fCTFont(font)
{
   //TODO: why don't I have asserts on parameters here?

   //Create attributed string with font and color.
   //TODO: Make code more general, this constructor is copy&paste.
   using MacOSX::Util::CFScopeGuard;

   const CFScopeGuard<CGColorSpaceRef> rgbColorSpace(CGColorSpaceCreateDeviceRGB());
   if (!rgbColorSpace.Get())
      throw std::runtime_error("TextLine: color space");

   Float_t rgba[] = {0.f, 0.f, 0.f, 1.f};
   GetTextColorForIndex(color, rgba[0], rgba[1], rgba[2], rgba[3]);
   const CGFloat cgRgba[] = {rgba[0], rgba[1], rgba[2], rgba[3]};

   const CFScopeGuard<CGColorRef> textColor(CGColorCreate(rgbColorSpace.Get(), cgRgba));
   //Not clear from docs, if textColor.Get() can be 0.
   
   CFStringRef keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
   CFTypeRef values[] = {font, textColor.Get()};

   Init(unichars, 2, keys, values);
}


//_________________________________________________________________
TextLine::~TextLine()
{
   CFRelease(fCTLine);
}


//_________________________________________________________________
void TextLine::GetBounds(UInt_t &w, UInt_t &h)const
{
   //The old 'fallback' version:
   CGFloat ascent = 0., descent = 0., leading = 0.;
   w = UInt_t(CTLineGetTypographicBounds(fCTLine, &ascent, &descent, &leading));
   h = UInt_t(ascent);// + descent + leading);
}


//_________________________________________________________________
void TextLine::GetAscentDescent(Int_t &asc, Int_t &desc)const
{
   //The old 'fallback' version:
   CGFloat ascent = 0., descent = 0., leading = 0.;
   CTLineGetTypographicBounds(fCTLine, &ascent, &descent, &leading);
   asc = Int_t(ascent);
   desc = Int_t(descent);
   //The new 'experimental':
   //with Core Text descent for a string '2' has some
   //quite big value, making all TText to be way too high.
   CFArrayRef runs = CTLineGetGlyphRuns(fCTLine);
   if (runs && CFArrayGetCount(runs) && fCTFont) {
      CTRunRef firstRun = static_cast<CTRunRef>(CFArrayGetValueAtIndex(runs, 0));
      CGRect box = BBoxForCTRun(fCTFont, firstRun);
      if (CGRectIsNull(box))
         return;
      
      for (CFIndex i = 1, e = CFArrayGetCount(runs); i < e; ++i) {
         CTRunRef run = static_cast<CTRunRef>(CFArrayGetValueAtIndex(runs, i));
         CGRect nextBox = BBoxForCTRun(fCTFont, run);
         if (CGRectIsNull(nextBox))
            return;
         box = CGRectUnion(box, nextBox);
      }

      asc = Int_t(TMath::Ceil(box.size.height) + box.origin.y);
      desc = Int_t(TMath::Abs(TMath::Floor(box.origin.y)));
   }
}


//_________________________________________________________________
void TextLine::Init(const char *textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values)
{
   using MacOSX::Util::CFScopeGuard;
   
   //Strong reference must be replaced with scope guards.
   const CFScopeGuard<CFDictionaryRef> stringAttribs(CFDictionaryCreate(kCFAllocatorDefault, (const void **)keys, (const void **)values,
                                                     nAttribs, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
   if (!stringAttribs.Get())
      throw std::runtime_error("TextLine: null attribs");

   const CFScopeGuard<CFStringRef> wrappedCString(CFStringCreateWithCString(kCFAllocatorDefault, textLine, kCFStringEncodingMacRoman));
   if (!wrappedCString.Get())
      throw std::runtime_error("TextLine: cstr wrapper");

   CFScopeGuard<CFAttributedStringRef> attributedString(CFAttributedStringCreate(kCFAllocatorDefault, wrappedCString.Get(), stringAttribs.Get()));
   fCTLine = CTLineCreateWithAttributedString(attributedString.Get());

   if (!fCTLine)
      throw std::runtime_error("TextLine: attrib string");
}

//_________________________________________________________________
void TextLine::Init(const std::vector<UniChar> &unichars, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values)
{
   using MacOSX::Util::CFScopeGuard;
   
   const CFScopeGuard<CFStringRef> wrappedUniString(CFStringCreateWithCharacters(kCFAllocatorDefault, &unichars[0], unichars.size()));
   const CFScopeGuard<CFDictionaryRef> stringAttribs(CFDictionaryCreate(kCFAllocatorDefault, (const void **)keys, (const void **)values,
                                                           nAttribs, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
   
   if (!stringAttribs.Get())
      throw std::runtime_error("TextLine: null attribs");

   if (!wrappedUniString.Get())
      throw std::runtime_error("TextLine: cstr wrapper");

   const CFScopeGuard<CFAttributedStringRef> attributedString(CFAttributedStringCreate(kCFAllocatorDefault,
                                                              wrappedUniString.Get(), stringAttribs.Get()));
   fCTLine = CTLineCreateWithAttributedString(attributedString.Get());

   if (!fCTLine)
      throw std::runtime_error("TextLine: attrib string");
}

//_________________________________________________________________
void TextLine::DrawLine(CGContextRef ctx)const
{
   assert(ctx != 0 && "DrawLine, ctx parameter is null");
   CTLineDraw(fCTLine, ctx);
}


//______________________________________________________________________________
void TextLine::DrawLine(CGContextRef ctx, Double_t x, Double_t y)const
{
   assert(ctx != 0 && "DrawLine, ctx parameter is null");

   CGContextSetAllowsAntialiasing(ctx, 1);
   UInt_t w = 0, h = 0;
   
   GetBounds(w, h);
   
   Double_t xc = 0., yc = 0.;   
   const UInt_t hAlign = UInt_t(gVirtualX->GetTextAlign() / 10);   
   switch (hAlign) {
   case 1:
      xc = 0.5 * w;
      break;
   case 2:
      break;
   case 3:
      xc = -0.5 * w;
      break;
   }

   const UInt_t vAlign = UInt_t(gVirtualX->GetTextAlign() % 10);
   switch (vAlign) {
   case 1:
      yc = 0.5 * h;
      break;
   case 2:
      break;
   case 3:
      yc = -0.5 * h;
      break;
   }
   
   CGContextSetTextPosition(ctx, 0., 0.);
   CGContextTranslateCTM(ctx, x, y);  
   CGContextRotateCTM(ctx, gVirtualX->GetTextAngle() * TMath::DegToRad());
   CGContextTranslateCTM(ctx, xc, yc);
   CGContextTranslateCTM(ctx, -0.5 * w, -0.5 * h);

   DrawLine(ctx);
}

//______________________________________________________________________________
void DrawTextLineNoKerning(CGContextRef ctx, CTFontRef font, const std::vector<UniChar> &text, Int_t x, Int_t y)
{
   typedef std::vector<CGSize>::size_type size_type;
   
   if (!text.size())//This can happen with ROOT's GUI.
      return;

   assert(ctx != 0 && "DrawTextLineNoKerning, ctx parameter is null");
   assert(font != 0 && "DrawTextLineNoKerning, font parameter is null");
   assert(text.size() && "DrawTextLineNoKerning, text parameter is an empty vector");

   std::vector<CGGlyph> glyphs(text.size());
   if (!CTFontGetGlyphsForCharacters(font, &text[0], &glyphs[0], text.size())) {
      ::Error("DrawTextLineNoKerning", "Font could not encode all Unicode characters in a text");
      return;
   }
   
   std::vector<CGSize> glyphAdvances(glyphs.size());
   CTFontGetAdvancesForGlyphs(font, horizontalFontOrientation, &glyphs[0], &glyphAdvances[0], glyphs.size());

   CGFloat currentX = x;  
   std::vector<CGPoint> glyphPositions(glyphs.size());
   glyphPositions[0].x = currentX;
   glyphPositions[0].y = y;

   for (size_type i = 1; i < glyphs.size(); ++i) {
      currentX += std::ceil(glyphAdvances[i - 1].width);
      glyphPositions[i].x = currentX;
      glyphPositions[i].y = y;
   }
   
   CTFontDrawGlyphs(font, &glyphs[0], &glyphPositions[0], glyphs.size(), ctx);
}

}//Quartz
}//ROOT
