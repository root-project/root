//Author: Timur Pocheptsov.

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

enum class FontSlant {
   regular,
   italic
};

enum class FontWeight {
   medium,
   bold
};

struct XLFDName {
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
