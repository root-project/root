//Author: Timur Pocheptsov.

#ifndef ROOT_FontCache
#define ROOT_FontCache

#include <map>

#include <ApplicationServices/ApplicationServices.h>

#ifndef ROOT_XLFDParser
#include "XLFDParser.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif

//////////////////////////////////////////////////////////////////
//                                                              //
// FontCache class:                                             //
// ROOT's GUI relies on TVirtualX to create and use fonts,      //
// fonts are referenced by integer identifiers.                 //
// Also, non-GUI graphics wants difference fonts.               //
// For the moment, this is quite lame implementation,           //
// which I will fix in a future (I promise! ;) ).               //
//                                                              //
//////////////////////////////////////////////////////////////////

namespace ROOT {
namespace MacOSX {
namespace Details {

class FontCache {
public:
   enum Details {
      nPadFonts = 13
   };

   FontCache();
   
   FontStruct_t LoadFont(const X11::XLFDName &xlfd);
   void UnloadFont(FontStruct_t font);

   unsigned GetTextWidth(FontStruct_t font, const char *text, int nChars);
   void GetFontProperties(FontStruct_t font, int &maxAscent, int &maxDescent);
   
   //Select the existing font or create a new one and select it.
   CTFontRef SelectFont(Font_t fontIndex, Float_t fontSize);
   
   //Typographical bounds (whatever it means),
   //for the current selected font and text.
   void GetTextBounds(UInt_t &w, UInt_t &h, const char *text)const;
   //
   double GetAscent()const;
   double GetDescent()const;
   double GetLeading()const;

private:

   CTFontRef SelectSymbolFont(Float_t fontSize);

   typedef Util::CFStrongReference<CTFontRef> CTFontGuard_t;
   
   //These are fonts for GUI. Weird map, as I can see now.
   std::map<CTFontRef, CTFontGuard_t> fLoadedFonts;

   //Fonts for TPad's graphics.
   typedef std::map<UInt_t, CTFontGuard_t> FontMap_t;

   FontMap_t fFonts[nPadFonts];
   CTFontRef fSelectedFont;
   
   FontCache(const FontCache &rhs) = delete;
   FontCache(FontCache &&rhs) = delete;//Make this explicit.
   FontCache &operator = (const FontCache &rhs) = delete;
   FontCache &operator = (FontCache &&rhs) = delete;//Make this explicit.
};

}
}
}

#endif
