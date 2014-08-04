// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   19/03/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#include <stdexcept>
#include <sstream>
#include <cassert>
#include <string>
#include <cmath>

#include "CocoaUtils.h"
#include "QuartzText.h"
#include "FontCache.h"
#include "TSystem.h"
#include "TError.h"
#include "TEnv.h"

namespace ROOT {
namespace MacOSX {
namespace Details {

namespace {

//ROOT uses indices for fonts. Indices are in the range [1 .. 15],
//12 is a symbol font (quite special thing, see the code below,
//15 is a "symbol italic" font - shear transformation is applied.

//TODO: actually, it's not good to assume I have these fonts for sure,
//find a better way to check the available fonts and search for the best
//match.

const CFStringRef fixedFontNames[FontCache::nPadFonts] =
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
                                      CFSTR("Symbol"),
                                      CFSTR("TimesNewRomanPSMT"),
                                      CFSTR("Wingdings"),
                                      CFSTR("Symbol-Italic")
                                     };


//______________________________________________________________________________
CTFontCollectionRef CreateFontCollection(const X11::XLFDName &/*xlfd*/)
{
   CTFontCollectionRef ctCollection = CTFontCollectionCreateFromAvailableFonts(0);
   if (!ctCollection)
      ::Error("CreateFontCollection", "CTFontCollectionCreateFromAvailableFonts failed");

   return ctCollection;
/*   CTFontCollectionRef ctCollection = 0;
   if (xlfd.fFamilyName == "*")
      ctCollection = CTFontCollectionCreateFromAvailableFonts(0);//Select all available fonts.
   else {
      //Create collection, using font descriptor?
      const Util::CFScopeGuard<CFStringRef> fontName(CFStringCreateWithCString(kCFAllocatorDefault, xlfd.fFamilyName.c_str(), kCFStringEncodingMacRoman));
      if (!fontName.Get()) {
         ::Error("CreateFontCollection", "CFStringCreateWithCString failed");
         return 0;
      }

      const Util::CFScopeGuard<CTFontDescriptorRef> fontDescriptor(CTFontDescriptorCreateWithNameAndSize(fontName.Get(), 0.));
      if (!fontDescriptor.Get()) {
         ::Error("CreateFontCollection", "CTFontDescriptorCreateWithNameAndSize failed");
         return 0;
      }

      Util::CFScopeGuard<CFMutableArrayRef> descriptors(CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks));
      if (!descriptors.Get()) {
         ::Error("CreateFontCollection", "CFArrayCreateMutable failed");
         return 0;
      }

      CFArrayAppendValue(descriptors.Get(), fontDescriptor.Get());
      ctCollection = CTFontCollectionCreateWithFontDescriptors(descriptors.Get(), 0);//Oh mama, so many code just to do this :(((
   }

   if (!ctCollection) {
      ::Error("CreateFontCollection", "No fonts are available for family %s", xlfd.fFamilyName.c_str());//WTF???
      return 0;
   }


   return ctCollection;*/
}

//______________________________________________________________________________
bool GetFamilyName(CTFontDescriptorRef fontDescriptor, std::vector<char> &name)
{
   //If success, this function returns a null-terminated string in a vector.
   assert(fontDescriptor != 0 && "GetFamilyName, parameter 'fontDescriptor' is null");

   name.clear();

   Util::CFScopeGuard<CFStringRef> cfFamilyName((CFStringRef)CTFontDescriptorCopyAttribute(fontDescriptor, kCTFontFamilyNameAttribute));
   if (const CFIndex cfLen = CFStringGetLength(cfFamilyName.Get())) {
      name.resize(cfLen + 1);//+ 1 for '\0'.
      if (CFStringGetCString(cfFamilyName.Get(), &name[0], name.size(), kCFStringEncodingMacRoman))
         return true;
   }

   return false;
}

#ifdef MAC_OS_X_VERSION_10_9

//______________________________________________________________________________
bool GetPostscriptName(CTFontDescriptorRef fontDescriptor, std::vector<char> &name)
{
   //If success, this function returns a null-terminated string in a vector.
   assert(fontDescriptor != 0 && "GetPostscriptName, parameter 'fontDescriptor' is null");

   name.clear();

   Util::CFScopeGuard<CFStringRef> cfFamilyName((CFStringRef)CTFontDescriptorCopyAttribute(fontDescriptor, kCTFontNameAttribute));

   if (const CFIndex cfLen = CFStringGetLength(cfFamilyName.Get())) {
      name.resize(cfLen + 1);//+ 1 for '\0'.
      if (CFStringGetCString(cfFamilyName.Get(), &name[0], name.size(), kCFStringEncodingMacRoman))
         return true;
   }

   return false;
}

#endif

//______________________________________________________________________________
void GetWeightAndSlant(CTFontDescriptorRef fontDescriptor, X11::XLFDName &newXLFD)
{
   //Let's ask for a weight and pixel size.
   const Util::CFScopeGuard<CFDictionaryRef> traits((CFDictionaryRef)CTFontDescriptorCopyAttribute(fontDescriptor, kCTFontTraitsAttribute));
   if (traits.Get()) {
      if (CFNumberRef symbolTraits = (CFNumberRef)CFDictionaryGetValue(traits.Get(), kCTFontSymbolicTrait)) {
         uint32_t val = 0;
         CFNumberGetValue(symbolTraits, kCFNumberIntType, &val);
         if (val & kCTFontItalicTrait)
            newXLFD.fSlant = X11::kFSItalic;
         else
            newXLFD.fSlant = X11::kFSRegular;

         if (val & kCTFontBoldTrait)
            newXLFD.fWeight = X11::kFWBold;
         else
            newXLFD.fWeight = X11::kFWMedium;
      }

      /*
      //The code below is wrong - using it, I can not identify bold or italic and always have
      //only medium/regular.
      if(CFNumberRef weight = (CFNumberRef)CFDictionaryGetValue(traits.Get(), kCTFontWeightTrait)) {
         double val = 0.;
         if (CFNumberGetValue(weight, kCFNumberDoubleType, &val))
            newXLFD.fWeight = val > 0. ? X11::kFWBold : X11::kFWMedium;
      }

      if(CFNumberRef slant = (CFNumberRef)CFDictionaryGetValue(traits.Get(), kCTFontSlantTrait)) {
         double val = 0.;
         if (CFNumberGetValue(slant, kCFNumberDoubleType, &val))
            newXLFD.fSlant = val > 0. ? X11::kFSItalic : X11::kFSRegular;
      }
      */
   }
}

//______________________________________________________________________________
void GetPixelSize(CTFontDescriptorRef fontDescriptor, X11::XLFDName &newXLFD)
{
   const Util::CFScopeGuard<CFNumberRef> size((CFNumberRef)CTFontDescriptorCopyAttribute(fontDescriptor, kCTFontSizeAttribute));
   if (size.Get()) {
      int pixelSize = 0;
      if(CFNumberIsFloatType(size.Get())) {
         double val = 0;
         CFNumberGetValue(size.Get(), kCFNumberDoubleType, &val);
         pixelSize = val;
      } else
         CFNumberGetValue(size.Get(), kCFNumberIntType, &pixelSize);

      if(pixelSize)
         newXLFD.fPixelSize = pixelSize;
   }
}

//_________________________________________________________________
void CreateXLFDString(const X11::XLFDName &xlfd, std::string &xlfdString)
{
    xlfdString = "-CoreText-"; //Fake foundry.
    xlfdString += xlfd.fFamilyName;

    if (xlfd.fWeight == X11::kFWBold)
        xlfdString += "-bold";
    else
        xlfdString += "-normal";

    if (xlfd.fSlant == X11::kFSItalic)
        xlfdString += "-i";
    else
        xlfdString += "-r";

    xlfdString += "-*-*"; //width, addstyle

    if (xlfd.fPixelSize) {
        std::ostringstream out;
        out<<xlfd.fPixelSize;
        xlfdString += "-";
        xlfdString += out.str();
    } else
        xlfdString += "-*";

    xlfdString += "-*-*-*-*-*-*-*-";//TODO: something more reasonable?
}

}

//_________________________________________________________________
FontCache::FontCache()
             : fSymbolFontRegistered(false)
{
   //XLFD name is not exactly PS name thus generating a warning with a new Core Text.
   fXLFDtoPostscriptNames["helvetica"] = "Helvetica";
   fXLFDtoPostscriptNames["courier"] = "Courier";
   fXLFDtoPostscriptNames["times"] = "Times-Roman";
}

//______________________________________________________________________________
FontStruct_t FontCache::LoadFont(const X11::XLFDName &xlfd)
{
   using Util::CFScopeGuard;
   using Util::CFStrongReference;

#ifdef MAC_OS_X_VERSION_10_9
   PSNameMap_t::const_iterator nameIt = fXLFDtoPostscriptNames.find(xlfd.fFamilyName);
   const std::string &psName = nameIt == fXLFDtoPostscriptNames.end() ? xlfd.fFamilyName : nameIt->second;
   const CFScopeGuard<CFStringRef> fontName(CFStringCreateWithCString(kCFAllocatorDefault, psName.c_str(), kCFStringEncodingMacRoman));
#else
   const CFScopeGuard<CFStringRef> fontName(CFStringCreateWithCString(kCFAllocatorDefault, xlfd.fFamilyName.c_str(), kCFStringEncodingMacRoman));
#endif

   const CFStrongReference<CTFontRef> baseFont(CTFontCreateWithName(fontName.Get(), xlfd.fPixelSize, 0), false);//false == do not retain

   if (!baseFont.Get()) {
      ::Error("FontCache::LoadFont", "CTFontCreateWithName failed for %s", xlfd.fFamilyName.c_str());
      return FontStruct_t();//Haha! Die ROOT, die!
   }

   CTFontSymbolicTraits symbolicTraits = CTFontSymbolicTraits();

   if (xlfd.fWeight == X11::kFWBold)
      symbolicTraits |= kCTFontBoldTrait;
   if (xlfd.fSlant == X11::kFSItalic)
      symbolicTraits |= kCTFontItalicTrait;

   if (symbolicTraits) {
      const CFStrongReference<CTFontRef> font(CTFontCreateCopyWithSymbolicTraits(baseFont.Get(), xlfd.fPixelSize, 0, symbolicTraits, symbolicTraits), false);//false == do not retain.
      if (font.Get()) {
         if (fLoadedFonts.find(font.Get()) == fLoadedFonts.end())
            fLoadedFonts[font.Get()] = font;

         return reinterpret_cast<FontStruct_t>(font.Get());
      }
   }

   if (fLoadedFonts.find(baseFont.Get()) == fLoadedFonts.end())
      fLoadedFonts[baseFont.Get()] = baseFont;

   return reinterpret_cast<FontStruct_t>(baseFont.Get());
}

//______________________________________________________________________________
void FontCache::UnloadFont(FontStruct_t font)
{
   CTFontRef fontRef = (CTFontRef)font;
   font_iterator fontIter = fLoadedFonts.find(fontRef);

   assert(fontIter != fLoadedFonts.end() && "Attempt to unload font, not created by font manager");

   fLoadedFonts.erase(fontIter);
}

//______________________________________________________________________________
char **FontCache::ListFonts(const X11::XLFDName &xlfd, int maxNames, int &count)
{
   typedef std::vector<char>::size_type size_type;

   count =  0;

   //Ugly, ugly code. I should "think different"!!!
   //To extract font names, I have to: create CFString, create font descriptor, create
   //CFArray, create CTFontCollection, that's a mess!!!
   //It's good I have my small and ugly RAII classes, otherwise the code will be
   //total trash and sodomy because of all possible cleanup actions.

   //First, create a font collection.
   const Util::CFScopeGuard<CTFontCollectionRef> collectionGuard(CreateFontCollection(xlfd));
   if (!collectionGuard.Get())
      return 0;

   Util::CFScopeGuard<CFArrayRef> fonts(CTFontCollectionCreateMatchingFontDescriptors(collectionGuard.Get()));
   if (!fonts.Get()) {
      ::Error("FontCache::ListFonts", "CTFontCollectionCreateMatchingFontDescriptors failed %s", xlfd.fFamilyName.c_str());
      return 0;
   }

   std::vector<char> xlfdData;
   //familyName is actually a null-terminated string.
   std::vector<char> familyName;
   X11::XLFDName newXLFD;
   std::string xlfdString;

   const CFIndex nFonts = CFArrayGetCount(fonts.Get());
   for (CFIndex i = 0; i < nFonts && count < maxNames; ++i) {
      CTFontDescriptorRef font = (CTFontDescriptorRef)CFArrayGetValueAtIndex(fonts.Get(), i);

      if (!GetFamilyName(font, familyName))
         continue;

      if (xlfd.fFamilyName != "*" && xlfd.fFamilyName != &familyName[0])
         continue;

      newXLFD.fFamilyName = &familyName[0];

      //If family name has '-', ROOT's GUI can not parse it correctly -
      //'-' is a separator in XLFD. Just skip this font (anyway, it wan not requested by GUI, only
      //listed by FontCache.
      if (newXLFD.fFamilyName.find('-') != std::string::npos)
         continue;

      GetWeightAndSlant(font, newXLFD);

      //Check weight and slant.
      if (xlfd.fWeight != X11::kFWAny && newXLFD.fWeight != xlfd.fWeight)
         continue;
      if (xlfd.fSlant != X11::kFSAny && newXLFD.fSlant != xlfd.fSlant)
         continue;

      if (xlfd.fPixelSize) {//Size was requested.
         GetPixelSize(font, newXLFD);
         //Core Text supports different font sizes.
         if (!newXLFD.fPixelSize)
            newXLFD.fPixelSize = xlfd.fPixelSize;
      }

#ifdef MAC_OS_X_VERSION_10_9
      //To avoid a warning from Core Text, save a mapping from a name seen by ROOT (family)
      //to a right postscript name (required by Core Text).

      //It's a null-terminated string:
      std::vector<char> postscriptName;
      if (GetPostscriptName(font, postscriptName)) {
         if (fXLFDtoPostscriptNames.find(&familyName[0]) == fXLFDtoPostscriptNames.end())
            fXLFDtoPostscriptNames[&familyName[0]] = &postscriptName[0];
      }
#endif


      //Ok, now lets create XLFD name, and place into list.
      CreateXLFDString(newXLFD, xlfdString);
      //
      xlfdData.insert(xlfdData.end(), xlfdString.begin(), xlfdString.end());
      xlfdData.push_back(0);//terminal 0.
      ++count;
   }

   //Setup array with string addresses.
   if (xlfdData.size()) {
      fFontLists.push_back(fDummyList);
      fFontLists.back().fStringData.swap(xlfdData);

      std::vector<char> &data = fFontLists.back().fStringData;
      std::vector<char *> &list = fFontLists.back().fList;

      list.push_back(&data[0]);
      for (size_type i = 1, e = data.size(); i < e; ++i) {
         if (!data[i] && i + 1 < e)
            list.push_back(&data[i + 1]);
      }

      return &list[0];
   } else
      return 0;
}

//______________________________________________________________________________
void FontCache::FreeFontNames(char **fontList)
{
   if (!fontList)
      return;

   for (std::list<FontList>::iterator it = fFontLists.begin(), eIt = fFontLists.end(); it != eIt; ++it) {
      if (fontList == &it->fList[0]) {
         fFontLists.erase(it);
         return;
      }
   }

   assert(0 && "FreeFontNames, unknown fontList");
}

//______________________________________________________________________________
unsigned FontCache::GetTextWidth(FontStruct_t font, const char *text, int nChars)
{
   typedef std::vector<CGSize>::size_type size_type;
   //
   CTFontRef fontRef = (CTFontRef)font;
   assert(fLoadedFonts.find(fontRef) != fLoadedFonts.end() && "Font was not created by font manager");

   //nChars is either positive, or negative (take all string).
   if (nChars < 0)
      nChars = std::strlen(text);

   std::vector<UniChar> unichars(text, text + nChars);

   //Extract glyphs for a text.
   std::vector<CGGlyph> glyphs(unichars.size());
   CTFontGetGlyphsForCharacters(fontRef, &unichars[0], &glyphs[0], unichars.size());

   //Glyps' advances for a text.
   std::vector<CGSize> glyphAdvances(glyphs.size());
   CTFontGetAdvancesForGlyphs(fontRef, kCTFontHorizontalOrientation, &glyphs[0], &glyphAdvances[0], glyphs.size());

   CGFloat textWidth = 0.;
   for (size_type i = 0, e = glyphAdvances.size(); i < e; ++i)
      textWidth += std::ceil(glyphAdvances[i].width);

   return textWidth;
}


//_________________________________________________________________
void FontCache::GetFontProperties(FontStruct_t font, int &maxAscent, int &maxDescent)
{
   CTFontRef fontRef = (CTFontRef)font;

   assert(fLoadedFonts.find(fontRef) != fLoadedFonts.end() && "Font was not created by font manager");

   try {
      maxAscent = int(CTFontGetAscent(fontRef) + 0.5) + 2;
      maxDescent = int(CTFontGetDescent(fontRef) + 0.5);
   } catch (const std::exception &) {
      throw;
   }
}


//_________________________________________________________________
CTFontRef FontCache::SelectFont(Font_t fontIndex, Float_t fontSize)
{
   fontIndex /= 10;

   if (fontIndex > nPadFonts || !fontIndex) {
      ::Warning("FontCache::SelectFont", "Font with index %d was requested", fontIndex);
      fontIndex = 3;//Select the Helvetica as default.
   } else
      fontIndex -= 1;

   if (fontIndex == 11 || fontIndex == 14)//Special case, our own symbol.ttf file.
      return SelectSymbolFont(fontSize, fontIndex);

   const UInt_t fixedSize = UInt_t(fontSize);
   font_map_iterator it = fFonts[fontIndex].find(fixedSize);

   if (it == fFonts[fontIndex].end()) {
      //Insert the new font.
      try {
         const CTFontGuard_t font(CTFontCreateWithName(fixedFontNames[fontIndex], fixedSize, 0), false);
         if (!font.Get()) {//With Apple's lame documentation it's not clear, if function can return 0.
            ::Error("FontCache::SelectFont", "CTFontCreateWithName failed for font %d", fontIndex);
            return 0;
         }

         fFonts[fontIndex][fixedSize] = font;//Insetion can throw.
         return fSelectedFont = font.Get();
      } catch (const std::exception &) {//Bad alloc.
         return 0;
      }
   }

   return fSelectedFont = it->second.Get();
}

//_________________________________________________________________
CTFontRef FontCache::SelectSymbolFont(Float_t fontSize, unsigned fontIndex)
{
   assert(fontIndex == 11 || fontIndex == 14 && "SelectSymbolFont, parameter fontIndex has invalid value");

   const UInt_t fixedSize = UInt_t(fontSize);
   font_map_iterator it = fFonts[fontIndex].find(fixedSize);//In ROOT, 11 is a font from symbol.ttf.

   if (it == fFonts[fontIndex].end()) {
      //This GetValue + Which I took from Olivier's code.
      const char * const fontDirectoryPath = gEnv->GetValue("Root.TTFontPath","$(ROOTSYS)/fonts");//This one I do not own.
      char * const fontFileName = gSystem->Which(fontDirectoryPath, "symbol.ttf", kReadPermission);//This must be deleted.

      const Util::ScopedArray<char> arrayGuard(fontFileName);

      if (!fontFileName || fontFileName[0] == 0) {
         ::Error("FontCache::SelectSymbolFont", "symbol.ttf file not found");
         return 0;
      }

      try {
         const Util::CFScopeGuard<CFStringRef> path(CFStringCreateWithCString(kCFAllocatorDefault, fontFileName, kCFURLPOSIXPathStyle));
         if (!path.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CFStringCreateWithCString failed");
            return 0;
         }

         const Util::CFScopeGuard<CFURLRef> fontURL(CFURLCreateWithFileSystemPath(kCFAllocatorDefault, path.Get(), kCFURLPOSIXPathStyle, false));
         if (!fontURL.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CFURLCreateWithFileSystemPath failed");
            return 0;
         }

         //Try to register this font.
         if (!fSymbolFontRegistered) {
            CFErrorRef err = 0;
            fSymbolFontRegistered = CTFontManagerRegisterFontsForURL(fontURL.Get(), kCTFontManagerScopeProcess, &err);
            if (!fSymbolFontRegistered) {
               ::Error("FontCache::SelectSymbolFont", "CTFontManagerRegisterFontsForURL failed");
               if (err)
                  CFRelease(err);
               return 0;
            }
         }

         const Util::CFScopeGuard<CFArrayRef> arr(CTFontManagerCreateFontDescriptorsFromURL(fontURL.Get()));
         if (!arr.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CTFontManagerCreateFontDescriptorsFromURL failed");
            return 0;
         }

         CTFontDescriptorRef fontDesc = (CTFontDescriptorRef)CFArrayGetValueAtIndex(arr.Get(), 0);

         const CGAffineTransform shearMatrix = {1., 0., 0.26794, 1., 0., 0.};//Yes, these are hardcoded values, taken from TPDF class.
         const CTFontGuard_t font(CTFontCreateWithFontDescriptorAndOptions(fontDesc, fixedSize,
                                                                           fontIndex == 11 ? &CGAffineTransformIdentity :
                                                                           &shearMatrix, kCTFontOptionsDefault), false);
         if (!font.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CTFontCreateWithFontDescriptor failed");
            return 0;
         }

         fFonts[fontIndex][fixedSize] = font;//This can throw.
         return fSelectedFont = font.Get();
      } catch (const std::exception &) {//Bad alloc.
         //RAII destructors should do their work.
         return 0;
      }
   }

   return fSelectedFont = it->second.Get();
}

//_________________________________________________________________
void FontCache::GetTextBounds(UInt_t &w, UInt_t &h, const char *text)const
{
   assert(fSelectedFont != 0 && "GetTextBounds: no font was selected");

   try {
      const Quartz::TextLine ctLine(text, fSelectedFont);
      ctLine.GetBounds(w, h);
      h += 2;
   } catch (const std::exception &) {
      throw;
   }
}

//_________________________________________________________________
void FontCache::GetTextBounds(UInt_t &w, UInt_t &h, const std::vector<UniChar> &unichars)const
{
   assert(fSelectedFont != 0 && "GetTextBounds: no font was selected");

   try {
      const Quartz::TextLine ctLine(unichars, fSelectedFont);
      ctLine.GetBounds(w, h);
      h += 2;
   } catch (const std::exception &) {
      throw;
   }
}

//_________________________________________________________________
double FontCache::GetAscent()const
{
   assert(fSelectedFont != 0 && "GetAscent, no font was selected");
   return CTFontGetAscent(fSelectedFont) + 1;
}

//_________________________________________________________________
double FontCache::GetAscent(const char *text)const
{
   assert(text != 0 && "GetAscent, parameter 'text' is null");
   assert(fSelectedFont != 0 && "GetAscent, no font was selected");

   try {
      const Quartz::TextLine ctLine(text, fSelectedFont);
      Int_t ascent = 0, descent = 0;
      ctLine.GetAscentDescent(ascent, descent);
      return ascent;
   } catch (const std::exception &) {
      throw;
   }
}

//_________________________________________________________________
double FontCache::GetAscent(const std::vector<UniChar> &unichars)const
{
   assert(fSelectedFont != 0 && "GetAscent, no font was selected");

   try {
      const Quartz::TextLine ctLine(unichars, fSelectedFont);
      Int_t ascent = 0, descent = 0;
      ctLine.GetAscentDescent(ascent, descent);
      return ascent;
   } catch (const std::exception &) {
      throw;
   }
}

//_________________________________________________________________
double FontCache::GetDescent()const
{
   assert(fSelectedFont != 0 && "GetDescent, no font was selected");
   return CTFontGetDescent(fSelectedFont) + 1;
}

//_________________________________________________________________
double FontCache::GetDescent(const char *text)const
{
   assert(text != 0 && "GetDescent, parameter 'text' is null");
   assert(fSelectedFont != 0 && "GetDescent, no font was selected");

   try {
      const Quartz::TextLine ctLine(text, fSelectedFont);
      Int_t ascent = 0, descent = 0;
      ctLine.GetAscentDescent(ascent, descent);
      return descent;
   } catch (const std::exception &) {
      throw;
   }
}

//_________________________________________________________________
double FontCache::GetDescent(const std::vector<UniChar> &unichars)const
{
   assert(fSelectedFont != 0 && "GetDescent, no font was selected");

   try {
      const Quartz::TextLine ctLine(unichars, fSelectedFont);
      Int_t ascent = 0, descent = 0;
      ctLine.GetAscentDescent(ascent, descent);
      return descent;
   } catch (const std::exception &) {
      throw;
   }
}

//_________________________________________________________________
double FontCache::GetLeading()const
{
   assert(fSelectedFont != 0 && "GetLeading, no font was selected");
   return CTFontGetLeading(fSelectedFont);
}


}//Details
}//MacOSX
}//ROOT
