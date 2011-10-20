// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#ifndef ROOT_IOSGraphicUtils
#define ROOT_IOSGraphicUtils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GraphicUtils                                                         //
//                                                                      //
// Aux. functions and classes.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace ROOT {
namespace iOS {
namespace GraphicUtils {

//Generic graphic utils.
void GetColorForIndex(Color_t colorIndex, Float_t &r, Float_t &g, Float_t &b);


//Encode object's ID (unsigned integer) as an RGB triplet.
class IDEncoder {
public:
   IDEncoder(UInt_t radix, UInt_t channelSize);
   
   Bool_t IdToColor(UInt_t objId, Float_t *rgb) const;
   UInt_t ColorToId(UInt_t r, UInt_t g, UInt_t b) const;
   
private:
   UInt_t FixValue(UInt_t val) const;

   const UInt_t fRadix;
   const UInt_t fRadix2;
   const UInt_t fChannelSize;
   const UInt_t fStepSize;
   const UInt_t fMaxID;
};

} //namespace GraphicUtils
} //namespace iOS
} //namespace ROOT

#endif
