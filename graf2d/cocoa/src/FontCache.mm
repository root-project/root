#define NDEBUG

#include <stdexcept>
#include <cassert>
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

//ROOT uses indices for fonts.
//Later, I'll find (I promise! ;) better
//way to map font indices to actual fonts
//(families, etc.) - I simply do not have any time now.

const int fmdNOfFonts = 13;
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
                                      CFSTR("TimesNewRomanPSMT")
                                     };


}

//_________________________________________________________________
FontCache::FontCache()
{
}

//______________________________________________________________________________
FontStruct_t FontCache::LoadFont(const X11::XLFDName &xlfd)
{
   using Util::CFScopeGuard;
   using Util::CFStrongReference;
   using X11::FontWeight;
   using X11::FontSlant;
   
   const CFScopeGuard<CFStringRef> fontName(CFStringCreateWithCString(kCFAllocatorDefault, xlfd.fFamilyName.c_str(), kCFStringEncodingMacRoman));
   const CFStrongReference<CTFontRef> baseFont(CTFontCreateWithName(fontName.Get(), xlfd.fPixelSize, 0), false);//false == do not retain
   
   if (!baseFont.Get()) {
      ::Error("FontCache::LoadFont", "CTFontCreateWithName failed for %s", xlfd.fFamilyName.c_str());
      return FontStruct_t();//Haha! Die ROOT, die!
   }
   
   CTFontSymbolicTraits symbolicTraits = CTFontSymbolicTraits();
   
   if (xlfd.fWeight == FontWeight::bold)
      symbolicTraits |= kCTFontBoldTrait;
   if (xlfd.fSlant == FontSlant::italic)
      symbolicTraits |= kCTFontItalicTrait;
      
   if (symbolicTraits) {
      const CFStrongReference<CTFontRef> font(CTFontCreateCopyWithSymbolicTraits(baseFont.Get(), xlfd.fPixelSize, nullptr, symbolicTraits, symbolicTraits), false);//false == do not retain.
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
   auto fontIter = fLoadedFonts.find(fontRef);

   assert(fontIter != fLoadedFonts.end() && "Attempt to unload font, not created by font manager");

   fLoadedFonts.erase(fontIter);
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

   //Instead of this, use CT funtion to request ascent/descent.
   try {
      const Quartz::TextLine textLine("LALALA", fontRef);
      textLine.GetAscentDescent(maxAscent, maxDescent);
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
      fontIndex = 1;
   }
   
   fontIndex -= 1;
   
   if (fontIndex == 11)//Special case, our own symbol.ttf file.
      return SelectSymbolFont(fontSize);
   
   const UInt_t fixedSize = UInt_t(fontSize);
   auto it = fFonts[fontIndex].find(fixedSize);
   
   if (it == fFonts[fontIndex].end()) {
      //Insert the new font.
      try {
         const CTFontGuard_t font(CTFontCreateWithName(fixedFontNames[fontIndex], fixedSize, 0), false);
         if (!font.Get()) {//With Apple's lame documentation it's not clear, if function can return 0.
            ::Error("FontCache::SelectFont", "CTFontCreateWithName failed for font %d", fontIndex);
            return nullptr;
         }
    
         fFonts[fontIndex][fixedSize] = font;//Insetion can throw.
         return fSelectedFont = font.Get();
      } catch (const std::exception &) {//Bad alloc.
         return nullptr;
      }
   }

   return fSelectedFont = it->second.Get();
}

//_________________________________________________________________
CTFontRef FontCache::SelectSymbolFont(Float_t fontSize)
{
   const UInt_t fixedSize = UInt_t(fontSize);
   auto it = fFonts[11].find(fixedSize);//In ROOT, 11 is a font from symbol.ttf.
   
   if (it == fFonts[11].end()) {
      //This GetValue + Which I took from Olivier's code.
      const char *fontDirectoryPath = gEnv->GetValue("Root.TTFontPath","$(ROOTSYS)/fonts");//This one I do not own.
      char *fontFileName = gSystem->Which(fontDirectoryPath, "symbol.ttf", kReadPermission);//This must be deleted.

      try {
         const Util::CFScopeGuard<CFStringRef> path(CFStringCreateWithCString(kCFAllocatorDefault, fontFileName, kCFURLPOSIXPathStyle));
         if (!path.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CFStringCreateWithCString failed");
            delete [] fontFileName;
            return nullptr;
         }
         
         const Util::CFScopeGuard<CFArrayRef> arr(CTFontManagerCreateFontDescriptorsFromURL(CFURLCreateWithFileSystemPath(kCFAllocatorDefault, path.Get(), kCFURLPOSIXPathStyle, false)));
         if (!arr.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CTFontManagerCreateFontDescriptorsFromURL failed");
            delete [] fontFileName;
            return nullptr;
         }

         CTFontDescriptorRef fontDesc = (CTFontDescriptorRef)CFArrayGetValueAtIndex(arr.Get(), 0);
         const CTFontGuard_t font(CTFontCreateWithFontDescriptor(fontDesc, fixedSize, 0), false);
         if (!font.Get()) {
            ::Error("FontCache::SelectSymbolFont", "CTFontCreateWithFontDescriptor failed");
            delete [] fontFileName;
            return nullptr;
         }

         fFonts[11][fixedSize] = font;//This can throw.
         return fSelectedFont = font.Get();
      } catch (const std::exception &) {//Bad alloc.
         return nullptr;
      }
   }

   return fSelectedFont = it->second.Get();
}


//_________________________________________________________________   
void FontCache::GetTextBounds(UInt_t &w, UInt_t &h, const char *text)const
{
   assert(fSelectedFont != nullptr && "GetTextBounds: no font was selected");
   
   try {
      const Quartz::TextLine ctLine(text, fSelectedFont);
      ctLine.GetBounds(w, h);
   } catch (const std::exception &) {
      throw;
   }
}


//_________________________________________________________________
double FontCache::GetAscent()const
{
   assert(fSelectedFont != nullptr && "GetAscent, no font was selected");
   return CTFontGetAscent(fSelectedFont);
}


//_________________________________________________________________
double FontCache::GetDescent()const
{
   assert(fSelectedFont != nullptr && "GetDescent, no font was selected");
   return CTFontGetDescent(fSelectedFont);
}

//_________________________________________________________________
double FontCache::GetLeading()const
{
   assert(fSelectedFont != nullptr && "GetLeading, no font was selected");
   return CTFontGetLeading(fSelectedFont);
}


}//Details
}//MacOSX
}//ROOT
