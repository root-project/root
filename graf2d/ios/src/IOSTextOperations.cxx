// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>
#include <utility>
#include <cstring>
#include <limits>
#include <string>

#include <CoreFoundation/CFAttributedString.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFBase.h>

#include <CoreText/CTStringAttributes.h>

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGColor.h>

#include "TVirtualX.h"
#include "TColor.h"
#include "TROOT.h"

#include "IOSTextOperations.h"
#include "IOSGraphicUtils.h"

namespace ROOT {
namespace iOS {

const CFStringRef fixedFontNames[FontManager::fmdNOfFonts] =
                                     {
                                      CFSTR("TimesNewRomanPS-ItalicMT"),
                                      CFSTR("TimesNewRomanPS-BoldMT"),
                                      CFSTR("TimesNewRomanPS-BoldItalicMT"),
                                      CFSTR("Helvetica"),
                                      CFSTR("Helvetica-Oblique"),
                                      CFSTR("Helvetica-Bold"),
                                      CFSTR("Helvetica-BoldOblique"),
                                      CFSTR("Courier"),
                                      CFSTR("Courier-Oblique"),
                                      CFSTR("Courier-Bold"),
                                      CFSTR("Courier-BoldOblique"),
                                      CFSTR("Helvetica"),
                                      CFSTR("TimesNewRomanPSMT")
                                     };

const char *cStrFontNames[FontManager::fmdNOfFonts] =
                              {
                               "TimesNewRomanPS-ItalicMT",
                               "TimesNewRomanPS-BoldMT",
                               "TimesNewRomanPS-BoldItalicMT",
                               "Helvetica",
                               "Helvetica-Oblique",
                               "Helvetica-Bold",
                               "Helvetica-BoldOblique",
                               "Courier",
                               "Courier-Oblique",
                               "Courier-Bold",
                               "Courier-BoldOblique",
                               "Helvetica",
                               "TimesNewRomanPSMT"
                              };

//_________________________________________________________________
CTLineGuard::CTLineGuard(const char *textLine, CTFontRef font)
                  : fCTLine(0)
{
   //Create attributed string with one attribue: the font.
   CFStringRef keys[] = {kCTFontAttributeName};
   CFTypeRef values[] = {font};

   Init(textLine, 1, keys, values);
}

//_________________________________________________________________
CTLineGuard::CTLineGuard(const char *textLine, CTFontRef font, Color_t /*color*/)
                  : fCTLine(0)
{
   //Create attributed string with font and color.
   Util::RefGuardGeneric<CGColorSpaceRef, CGColorSpaceRelease> rgbColorSpace(CGColorSpaceCreateDeviceRGB());
   if (!rgbColorSpace.Get())
      throw std::runtime_error("CTLineGuard: color space");

   CGFloat rgba[] = {0.f, 0.f, 0.f, 1.f};
   Float_t r = 0.f, g = 0.f, b = 0.f;
   GraphicUtils::GetColorForIndex(gVirtualX->GetTextColor(), r, g, b);
   rgba[0] = r; rgba[1] = g; rgba[2] = b;

   Util::RefGuardGeneric<CGColorRef, CGColorRelease> textColor(CGColorCreate(rgbColorSpace.Get(), rgba));
   //Not clear from docs, if textColor.Get() can be 0.

   CFStringRef keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
   CFTypeRef values[] = {font, textColor.Get()};

   Init(textLine, 2, keys, values);
}

//_________________________________________________________________
CTLineGuard::CTLineGuard(const char *textLine, CTFontRef font, const std::vector<UniChar> &symbolMap)
                  : fCTLine(0)
{
   //Create attributed string with font and color.
   Util::RefGuardGeneric<CGColorSpaceRef, CGColorSpaceRelease> rgbColorSpace(CGColorSpaceCreateDeviceRGB());
   if (!rgbColorSpace.Get())
      throw std::runtime_error("CTLineGuard: color space");

   CGFloat rgba[] = {0.f, 0.f, 0.f, 1.f};
   Float_t r = 0.f, g = 0.f, b = 0.f;
   GraphicUtils::GetColorForIndex(gVirtualX->GetTextColor(), r, g, b);
   rgba[0] = r; rgba[1] = g; rgba[2] = b;

   Util::RefGuardGeneric<CGColorRef, CGColorRelease> textColor(CGColorCreate(rgbColorSpace.Get(), rgba));
   const unsigned length = std::strlen(textLine);
   std::vector<UniChar> convertedLine(length);
   for (unsigned i = 0; i < length; ++i)
      convertedLine[i] = symbolMap[(unsigned char)textLine[i]];

   CFStringRef keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
   CFTypeRef values[] = {font, textColor.Get()};

   Init(convertedLine, 2, keys, values);
}

//_________________________________________________________________
CTLineGuard::~CTLineGuard()
{
   CFRelease(fCTLine);
}

//_________________________________________________________________
void CTLineGuard::GetBounds(UInt_t &w, UInt_t &h)const
{
   CGFloat ascent = 0.f, descent = 0.f, leading = 0.f;
   w = UInt_t(CTLineGetTypographicBounds(fCTLine, &ascent, &descent, &leading));
   h = UInt_t(ascent);// + descent + leading);
}

//_________________________________________________________________
void CTLineGuard::Init(const char *textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values)
{
   Util::RefGuard<CFDictionaryRef> stringAttribs(
                                                 CFDictionaryCreate(kCFAllocatorDefault,
                                                                    (const void **)keys,
                                                                    (const void **)values,
                                                                    nAttribs,
                                                                    &kCFTypeDictionaryKeyCallBacks,
                                                                    &kCFTypeDictionaryValueCallBacks)
                                                );

   if (!stringAttribs.Get())
      throw std::runtime_error("CTLineGuard: null attribs");

   Util::RefGuard<CFStringRef> wrappedCString(CFStringCreateWithCString(kCFAllocatorDefault, textLine, kCFStringEncodingMacRoman));
   if (!wrappedCString.Get())
      throw std::runtime_error("CTLineGuard: cstr wrapper");

   Util::RefGuard<CFAttributedStringRef> attributedString(CFAttributedStringCreate(kCFAllocatorDefault, wrappedCString.Get(), stringAttribs.Get()));
   fCTLine = CTLineCreateWithAttributedString(attributedString.Get());

   if (!fCTLine)
      throw std::runtime_error("CTLineGuard: attrib string");
}

//_________________________________________________________________
void CTLineGuard::Init(const std::vector<UniChar> &textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values)
{
   Util::RefGuard<CFDictionaryRef> stringAttribs(
                                                 CFDictionaryCreate(kCFAllocatorDefault,
                                                                    (const void **)keys,
                                                                    (const void **)values,
                                                                    nAttribs,
                                                                    &kCFTypeDictionaryKeyCallBacks,
                                                                    &kCFTypeDictionaryValueCallBacks)
                                                );

   if (!stringAttribs.Get())
      throw std::runtime_error("CTLineGuard: null attribs");

   Util::RefGuard<CFStringRef> wrappedCString(CFStringCreateWithCharacters(kCFAllocatorDefault, &textLine[0], textLine.size()));
   if (!wrappedCString.Get())
      throw std::runtime_error("CTLineGuard: cstr wrapper");

   Util::RefGuard<CFAttributedStringRef> attributedString(CFAttributedStringCreate(kCFAllocatorDefault, wrappedCString.Get(), stringAttribs.Get()));
   fCTLine = CTLineCreateWithAttributedString(attributedString.Get());

   if (!fCTLine)
      throw std::runtime_error("CTLineGuard: attrib string");
}

//_________________________________________________________________
FontManager::FontManager()
               : fSelectedFont(0)
{
}

//_________________________________________________________________
FontManager::~FontManager()
{
   for (UInt_t i = 0; i < fmdNOfFonts; ++i)
      for (FontMapIter_t it = fFonts[i].begin(); it != fFonts[i].end(); ++it)
         CFRelease(it->second);
}

//_________________________________________________________________
CTFontRef FontManager::SelectFont(Font_t fontIndex, Float_t fontSize)
{
   fontIndex /= 10;

   if (fontIndex > fmdNOfFonts || !fontIndex)
      throw std::runtime_error("SelectFont: index");

   fontIndex -= 1;

   if (fontIndex == 11 && !fSymbolMap.size())
      InitSymbolMap();

   const UInt_t fixedSize = UInt_t(fontSize);
   FontMapIter_t it = fFonts[fontIndex].find(fixedSize);

   if (it == fFonts[fontIndex].end()) {
      //Insert the new font.
      Util::RefGuard<CTFontRef> font(CTFontCreateWithName(fixedFontNames[fontIndex], fixedSize, 0));
      if (!font.Get()) //With Apple's lame documentation it's not clear, if function can return 0.
         throw std::runtime_error(std::string("SelectFont: create") + cStrFontNames[fontIndex]);

      fFonts[fontIndex][fixedSize] = font.Get();
      return fSelectedFont = font.Release();
   }

   return fSelectedFont = it->second;
}

//_________________________________________________________________
void FontManager::GetTextBounds(UInt_t &w, UInt_t &h, const char *text)const
{
   if (!fSelectedFont)
      throw std::runtime_error("GetTextBounds: font not selected");

   Font_t fontIndex = gVirtualX->GetTextFont() / 10 - 1;

   if (fontIndex == 11) {
      CTLineGuard ctLine(text, fSelectedFont, fSymbolMap);
      ctLine.GetBounds(w, h);
   } else {
      CTLineGuard ctLine(text, fSelectedFont);
      ctLine.GetBounds(w, h);
   }
}

//_________________________________________________________________
Double_t FontManager::GetAscent()const
{
   if (!fSelectedFont)
      throw std::runtime_error("GetAscent");

   return CTFontGetAscent(fSelectedFont);
}

//_________________________________________________________________
Double_t FontManager::GetDescent()const
{
   if (!fSelectedFont)
      throw std::runtime_error("GetDescent");

   return CTFontGetDescent(fSelectedFont);
}

//_________________________________________________________________
Double_t FontManager::GetLeading()const
{
   if (!fSelectedFont)
      throw std::runtime_error("GetLeading");

   return CTFontGetLeading(fSelectedFont);
}

//_________________________________________________________________
void FontManager::InitSymbolMap()
{
   fSymbolMap.clear();
   fSymbolMap.resize(1 << std::numeric_limits<unsigned char>::digits, 0);

   fSymbolMap[97]  = 0x3B1; //alpha
   fSymbolMap[98]  = 0x3B2; //beta
   fSymbolMap[103] = 0x3B3; //gamma
   fSymbolMap[100] = 0x3B4; //delta
   fSymbolMap[206] = 0x3F5; //epsilon
   fSymbolMap[122] = 0x3B6; //zeta
   fSymbolMap[104] = 0x3B7; //eta
   fSymbolMap[113] = 0x3B8; //theta
   fSymbolMap[105] = 0x3B9; //iota
   fSymbolMap[107] = 0x3BA; //kappa
   fSymbolMap[108] = 0x3BB; //lambda
   fSymbolMap[109] = 0x3BC; //mu
   fSymbolMap[110] = 0x3BD; //nu
   fSymbolMap[120] = 0x3BE; //xi
   fSymbolMap[111] = 0x3BF; //omicron
   fSymbolMap[112] = 0x3C0; //pi
   fSymbolMap[114] = 0x3C1; //rho
   fSymbolMap[115] = 0x3C3; //sigma
   fSymbolMap[116] = 0x3C4; //tau
   fSymbolMap[117] = 0x3C5; //upsilon
   fSymbolMap[102] = 0x3C6; //phi
   fSymbolMap[99]  = 0x3C7; //chi
   fSymbolMap[121] = 0x3C8; //psi
   fSymbolMap[119] = 0x3C9; //omega

   fSymbolMap[65] = 0x391; //Alpha
   fSymbolMap[66] = 0x392; //Beta
   fSymbolMap[71] = 0x393; //Gamma
   fSymbolMap[68] = 0x394; //Delta
   fSymbolMap[69] = 0x395; //Epsilon
   fSymbolMap[90] = 0x396; //Zeta
   fSymbolMap[72] = 0x397; //Eta
   fSymbolMap[81] = 0x398; //Theta
   fSymbolMap[73] = 0x399; //Iota
   fSymbolMap[75] = 0x39A; //Kappa
   fSymbolMap[76] = 0x39B; //Lambda
   fSymbolMap[77] = 0x39C; //Mu
   fSymbolMap[78] = 0x39D; //Nu
   fSymbolMap[88] = 0x39E; //Xi
   fSymbolMap[79] = 0x39F; //Omicron
   fSymbolMap[80] = 0x3A0; //Pi
   fSymbolMap[82] = 0x3A1; //Rho
   fSymbolMap[83] = 0x3A3; //Sigma
   fSymbolMap[84] = 0x3A4; //Tau
   fSymbolMap[85] = 0x3A5; //Upsilon
   fSymbolMap[70] = 0x3A6; //Phi
   fSymbolMap[67] = 0x3A7; //Chi
   fSymbolMap[89] = 0x3A8; //Psi
   fSymbolMap[87] = 0x3A9; //Omega

   fSymbolMap[101] = 0x3B5; //varepsilon
   fSymbolMap[74]  = 0x3D1; //vartheta
   fSymbolMap[86]  = 0x3C2; //varsigma
   fSymbolMap[161] = 0x3D2; //varUpsilon
   fSymbolMap[106] = 0x3D5; //varphi???
   fSymbolMap[118] = 0x3D6; //varomega?

   fSymbolMap[167] = 0x2663;
   fSymbolMap[195] = 0x2118;
   fSymbolMap[163] = 0x2264;
   fSymbolMap[187] = 0x2248;
   fSymbolMap[206] = 0x2208;
   fSymbolMap[201] = 0x2283;
   fSymbolMap[199] = 0x2229;
   fSymbolMap[211] = 0xA9;
   fSymbolMap[212] = 0x2122;
   fSymbolMap[180] = 0xD7;
   fSymbolMap[183] = 0x2022;
   fSymbolMap[166] = 0x192;
   fSymbolMap[178] = 0x2033;
   fSymbolMap[231] = 0x7C;
   fSymbolMap[232] = 0x23A9;
   fSymbolMap[175] = 0x2193;
   fSymbolMap[171] = 0x2194;
   fSymbolMap[223] = 0x21D3;
   fSymbolMap[219] = 0x21D4;
   fSymbolMap[234] = 0x23AA;
   fSymbolMap[104] = 0x127;

   fSymbolMap[168] = 0x2666;
   fSymbolMap[192] = 0x2135;
   fSymbolMap[179] = 0x2265;
   fSymbolMap[185] = 0x2260;
   fSymbolMap[207] = 0x2209;
   fSymbolMap[205] = 0x2286;
   fSymbolMap[200] = 0x222A;
   fSymbolMap[227] = 0xA9;
   fSymbolMap[228] = 0x2122;
   fSymbolMap[184] = 0xF7;
   fSymbolMap[176] = 0xB0;
   fSymbolMap[165] = 0x221E;
   fSymbolMap[208] = 0x2220;
   fSymbolMap[189] = 0x7C;
   fSymbolMap[230] = 0x23A7;
   fSymbolMap[172] = 0x2190;
   fSymbolMap[196] = 0x2297;
   fSymbolMap[220] = 0x21D0;
   fSymbolMap[213] = 0x220F;

   fSymbolMap[169] = 0x2665;
   fSymbolMap[193] = 0x2111;
   fSymbolMap[225] = 0x3008;
   fSymbolMap[186] = 0x2261;
   fSymbolMap[204] = 0x2282;
   fSymbolMap[202] = 0x2287;
   fSymbolMap[217] = 0x2227;
   fSymbolMap[210] = 0xAE;
   fSymbolMap[197] = 0xC5; //no need
   fSymbolMap[177] = 0xB1;
   fSymbolMap[188] = 0x2026;
   fSymbolMap[209] = 0x2207;
   fSymbolMap[191] = 0x21B5;
   fSymbolMap[190] = 0x2015;//WRONG
   fSymbolMap[236] = 0x23A7;
   fSymbolMap[173] = 0x2191;
   fSymbolMap[197] = 0x2295;
   fSymbolMap[221] = 0x21D1;
   fSymbolMap[229] = 0x2211;
   fSymbolMap[34] = 0x2200;

   fSymbolMap[170] = 0x2660;
   fSymbolMap[194] = 0x211C;
   fSymbolMap[241] = 0x3009;
   fSymbolMap[181] = 0x221D;
   fSymbolMap[203] = 0x2284;
   fSymbolMap[198] = 0x2205;
   fSymbolMap[218] = 0x2228;
   fSymbolMap[226] = 0xAE;
   fSymbolMap[229] = 0xE5; //no need
   fSymbolMap[164] = 0x2044;
   fSymbolMap[215] = 0x22C5;
   fSymbolMap[182] = 0x2202;
   fSymbolMap[216] = 0xAC;
   fSymbolMap[237] = 0x23A8;
   fSymbolMap[235] = 0x23A3;
   fSymbolMap[174] = 0x2192;
   fSymbolMap[214] = 0x221A;
   fSymbolMap[222] = 0x21D2;
   fSymbolMap[242] = 0x222B;
   fSymbolMap[36] = 0x2203;
}

}//namespace iOS
}//namespace ROOT
