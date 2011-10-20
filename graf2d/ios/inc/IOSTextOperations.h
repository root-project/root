// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TextOperations
#define ROOT_TextOperations

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IOSTextOperations                                                    //
//                                                                      //
// Font management for iOS and Core Text.                               //
// To be extended or completely changed in a future.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <map>

#include <CoreText/CTFont.h>
#include <CoreText/CTLine.h>

#ifndef ROOT_IOSResourceManagement
#include "IOSResourceManagement.h"
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace ROOT {
namespace iOS {

class CTLineGuard : public Util::NonCopyable {
   friend class Painter;

public:
   CTLineGuard(const char *textLine, CTFontRef font);
   CTLineGuard(const char *textLine, CTFontRef font, Color_t color);
   CTLineGuard(const char *textLine, CTFontRef font, const std::vector<UniChar> &symbolMap);
   ~CTLineGuard();
   
   void GetBounds(UInt_t &w, UInt_t &h)const;
   
private:

   void Init(const char *textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values);
   void Init(const std::vector<UniChar> &textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values);

   CTLineRef fCTLine; //Core Text line, created from Attributed string.
};

class FontManager : public Util::NonCopyable {
public:
   enum FontManagerDefaults {
      fmdNOfFonts = 13
   };

   FontManager();
   ~FontManager();

   //Select the existing font or create a new one and select it.
   CTFontRef SelectFont(Font_t fontIndex, Float_t fontSize);
   
   //Typographical bounds (whatever it means),
   //for the current selected font and text.
   void GetTextBounds(UInt_t &w, UInt_t &h, const char *text)const;
   //
   double GetAscent()const;
   double GetDescent()const;
   double GetLeading()const;
   
   const std::vector<UniChar> &GetSymbolMap()const
   {
      return fSymbolMap;
   }

private:
   typedef std::map<UInt_t, CTFontRef> FontMap_t;
   typedef FontMap_t::iterator FontMapIter_t;
   typedef FontMap_t::const_iterator FontMapConstIter_t;

   FontMap_t fFonts[fmdNOfFonts];
   CTFontRef fSelectedFont;
   
   std::vector<UniChar> fSymbolMap;
   
   void InitSymbolMap();
};

}//namespace iOS
}//namespace ROOT


#endif
