// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   19/03/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_FontCache
#define ROOT_FontCache

#include <vector>
#include <list>
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
//                                                              //
//////////////////////////////////////////////////////////////////

namespace ROOT {
namespace MacOSX {
namespace Details {

class FontCache {
public:
   enum Details {
      nPadFonts = 15
   };

   FontCache();
   
   FontStruct_t LoadFont(const X11::XLFDName &xlfd);
   void UnloadFont(FontStruct_t font);
   
   char **ListFonts(const X11::XLFDName &xlfd, int maxNames, int &count);
   void FreeFontNames(char **fontList);

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

   //We have "two symbolic" fonts, both of them use the same symbol.ttf (index 11),
   //but the second one (index
   CTFontRef SelectSymbolFont(Float_t fontSize, unsigned fontIndex);

   typedef Util::CFStrongReference<CTFontRef> CTFontGuard_t;
   
   //These are fonts for GUI. Weird map, as I can see now.
   std::map<CTFontRef, CTFontGuard_t> fLoadedFonts;
   typedef std::map<CTFontRef, CTFontGuard_t>::iterator font_iterator;
   typedef std::map<CTFontRef, CTFontGuard_t>::const_iterator const_font_iterator;

   //Fonts for TPad's graphics.
   typedef std::map<UInt_t, CTFontGuard_t> FontMap_t;
   typedef FontMap_t::iterator font_map_iterator;
   typedef FontMap_t::const_iterator const_font_map_iterator;

   FontMap_t fFonts[nPadFonts];
   CTFontRef fSelectedFont;

   //FontList can be requested by TGCocoa::ListFonts,
   //the return value is char **, and later it's freed by
   //TGCocoa::FreeFontNames, again using char **.
   //In my case, I have to somehow map char ** to two 
   //data sets - char ** itself + real strings, whose
   //addresses are in char **.
   //fList, after it's filled and returned by TGCocoa, 
   //is immutable, so later I can find this FontList
   //comparing char ** and &fList[0].
   struct FontList {
      std::vector<char *> fList;
      std::vector<char> fStringData;
   };

   std::list<FontList> fFontLists;//list of "lists" of fonts :)
   FontList fDummyList;   

   FontCache(const FontCache &rhs);
   FontCache &operator = (const FontCache &rhs);

   bool fSymbolFontRegistered;
};

}
}
}

#endif
