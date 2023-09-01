// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   28/11/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_X11Colors
#define ROOT_X11Colors

/////////////////////////////////////////////////////////////////
//                                                             //
// ROOT's GUI depends and relies on X11 XParseColors.          //
// Color can be specified by name (found in X11 rgb.txt)       //
// or as RGB triplet: #rgb #rrggbb #rrrgggbbb #rrrrggggbbbb.   //
//                                                             //
/////////////////////////////////////////////////////////////////

#include <map>

#include <ApplicationServices/ApplicationServices.h>

#include "GuiTypes.h"
#include "TString.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

class ColorParser {
public:
   ColorParser();

   bool ParseColor(const TString &colorString, ColorStruct_t &color)const;

private:
   bool ParseRGBTriplet(const TString &rgb, ColorStruct_t &color)const;
   bool LookupColorByName(const TString &colorName, ColorStruct_t &color)const;

   struct RGB_t {
      RGB_t()
         : fRed(0), fGreen(0), fBlue(0)
      {
      }
      RGB_t(unsigned r, unsigned g, unsigned b)
         : fRed(r), fGreen(g), fBlue(b)
      {
      }
      unsigned fRed;
      unsigned fGreen;
      unsigned fBlue;
   };

   std::map<TString, RGB_t> fX11RGB;//X11's rgb.txt
   typedef std::map<TString, RGB_t> rgb_map;
   typedef rgb_map::const_iterator const_rgb_iterator;
};

void PixelToRGB(Pixel_t pixelColor, CGFloat *rgb);
void PixelToRGB(Pixel_t pixelColor, unsigned char *rgb);

}//X11
}//MacOSX
}//ROOT

#endif
