// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#include "TAttMarker.h"
#include "TVirtualX.h"
#include "TColor.h"
#include "TROOT.h"

#include "IOSGraphicUtils.h"

namespace ROOT {
namespace iOS {
namespace GraphicUtils {

//______________________________________________________________________________
void GetColorForIndex(Color_t colorIndex, Float_t &r, Float_t &g, Float_t &b)
{
   if (const TColor *color = gROOT->GetColor(colorIndex))
      color->GetRGB(r, g, b);
}

//IDEncoder.
//______________________________________________________________
IDEncoder::IDEncoder(UInt_t radix, UInt_t channelSize)
            : fRadix(radix),
              fRadix2(radix * radix),
              fChannelSize(channelSize),
              fStepSize(channelSize / (radix - 1)),
              fMaxID(radix * radix * radix)
{
}

//______________________________________________________________
Bool_t IDEncoder::IdToColor(UInt_t id, Float_t *rgb) const
{
   if (id >= fMaxID)
      return kFALSE;

   const UInt_t red = id / fRadix2;
   const UInt_t green = (id - red * fRadix2) / fRadix;
   const UInt_t blue = (id - red * fRadix2 - green * fRadix) % fRadix;
   
   rgb[0] = red * fStepSize / Float_t(fChannelSize);
   rgb[1] = green * fStepSize / Float_t(fChannelSize);
   rgb[2] = blue * fStepSize / Float_t(fChannelSize);
   
   return kTRUE;
}

//______________________________________________________________
UInt_t IDEncoder::ColorToId(UInt_t r, UInt_t g, UInt_t b) const
{
   const UInt_t red   = FixValue(r);
   const UInt_t green = FixValue(g);
   const UInt_t blue  = FixValue(b);
   
   return fRadix2 * red + fRadix * green + blue;
}

//______________________________________________________________
UInt_t IDEncoder::FixValue(UInt_t val) const
{
   const UInt_t orig = val / fStepSize;
   
   if (orig * fStepSize != val) {
      const UInt_t top = (orig + 1) * fStepSize - val;
      const UInt_t bottom = val - orig * fStepSize;

      if (top > bottom || orig + 1 >= fRadix)
         return orig;
         
      return orig + 1;
   } else
      return orig;
}

}//namespace GraphicUtils
}//namespace iOS
}//namespace ROOT
