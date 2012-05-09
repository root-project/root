// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   2/03/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XLFDParser
#define ROOT_XLFDParser

#include <string>

////////////////////////////////////////////////////////////////////////
//                                                                    //
// XLDF parser, very simple implementation, used by GUI only.         //
//                                                                    //
////////////////////////////////////////////////////////////////////////

namespace ROOT {
namespace MacOSX {
namespace X11 {//X11 emulation.

enum FontSlant {
   kFSAny, //For '*' wildcard in xlfd string.
   kFSRegular,
   kFSItalic
};

enum FontWeight {
   kFWAny, //For '*' wildcard in xlfd string.
   kFWMedium,
   kFWBold
};

struct XLFDName {
   XLFDName();
   //foundry *
   std::string fFamilyName;
   FontWeight fWeight;
   FontSlant fSlant;
   //width  *
   //addstyle *
   unsigned fPixelSize;
   //points *
   //horiz *
   //vert *
   //spacing *
   //avgwidth *
   std::string fRgstry;
   std::string fEncoding;
};

bool ParseXLFDName(const std::string &xlfdName, XLFDName &dst);

}//X11
}//MacOSX
}//ROOT

#endif
