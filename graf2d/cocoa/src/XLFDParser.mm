// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   2/03/2012

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
#include <cctype>

#include "XLFDParser.h"
#include "TError.h"

//
// I did not find any formal description of what XLFD name can be.
// The first version of this code was quite strict - it expected all
// components to be in place, digits must be digits. But after using
// ROOT's GUI for some time, I noticed that ROOT can use font name like
// *, "fixed", "-*-" or something like this. In this case I have to set
// some "default" or "wildcard" names.
//

namespace ROOT {
namespace MacOSX {
namespace X11 {

namespace {

typedef std::string::size_type size_type;

//______________________________________________________________________________
template<class T>
void StringToInt(const std::string &str, const std::string &componentName, T &num)
{
   for (size_type i = 0, e = str.length(); i < e; ++i) {
      const char symbol = str[i];
      if (!std::isdigit(symbol))
         throw std::runtime_error("bad symbol while converting component " + componentName + " into number");
   }

   std::istringstream in(str);
   in>>num;
}

//______________________________________________________________________________
size_type GetXLFDNameComponentAsString(const std::string &name, const std::string & componentName,
                                       size_type pos, std::string &component)
{
   const size_type length = name.length();
   if (pos + 1 >= length)
      throw std::runtime_error("Unexpected end of name while parsing " + componentName);

   //Starting symbol must be '-'.
   if (name[pos] != '-')
      throw std::runtime_error("Component " + componentName + " must start from '-'");

   const size_type start = ++pos;
   ++pos;
   while (pos < length && name[pos] != '-')
      ++pos;

   if (pos - start)
      component = name.substr(start, pos - start);
   else
      component = "";

   return pos;
}


//______________________________________________________________________________
template<class T>
size_type GetXLFDNameComponentAsInteger(const std::string &name,
                                        const std::string &componentName,
                                        size_type pos, T &component)
{
   std::string num;
   pos = GetXLFDNameComponentAsString(name, componentName, pos, num);
   StringToInt(num, componentName, component);

   return pos;
}


//______________________________________________________________________________
size_type ParseFoundry(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignore foundry.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "foundry", pos, dummy);
}


//______________________________________________________________________________
size_type ParseFamilyName(const std::string &name, size_type pos, XLFDName &dst)
{
   const size_type tokenEnd = GetXLFDNameComponentAsString(name, "family name", pos, dst.fFamilyName);

   //This is a "special case": ROOT uses it's own symbol.ttf, but
   //I can not use it to render text in a GUI, and also I can not use
   //Apple's system symbol font - it can not encode all symbols ROOT wants.
   if (dst.fFamilyName == "symbol")
      dst.fFamilyName = "helvetica";

   return tokenEnd;
}


//______________________________________________________________________________
size_type ParseWeight(const std::string &name, size_type pos, XLFDName &dst)
{
   //Weight can be an integer, can be a word, can be a combination of a word
   //and integer.
   std::string weight;
   pos = GetXLFDNameComponentAsString(name, "weight", pos, weight);

   if (weight == "*")
      dst.fWeight = kFWAny;
   else if (weight == "bold")
      dst.fWeight = kFWBold;
   else
      dst.fWeight = kFWMedium;

   return pos;
}


//______________________________________________________________________________
size_type ParseSlant(const std::string &name, size_type pos, XLFDName &dst)
{
   //Slant can be regular or italic now.
   std::string slant;
   pos = GetXLFDNameComponentAsString(name, "slant", pos, slant);

   //Can be 'r', 'R', 'i', 'I', 'o', 'O', '*'.
   if (slant == "*")
      dst.fSlant = kFSAny;
   else if (slant == "i" || slant == "I" || slant == "o" || slant == "O")
      dst.fSlant = kFSItalic;
   else
      dst.fSlant = kFSRegular;

   return pos;
}

//______________________________________________________________________________
size_type ParseSetwidth(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Setwidth is ignored now.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "setwidth", pos, dummy);
}

//______________________________________________________________________________
size_type ParseAddstyle(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignored at the moment.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "addstyle", pos, dummy);
}

//______________________________________________________________________________
size_type ParsePixelSize(const std::string &name, size_type pos, XLFDName &dst)
{
   //First, try to parse as string. It can be '*' == 'any size'.
   //In the first version it was more strict - throwing and exception.
   std::string dummy;

   size_type endOfSize = GetXLFDNameComponentAsString(name, "pixel size", pos, dummy);
   if (dummy == "*") {
      //Aha, ignore size.
      dst.fPixelSize = 0;
      return endOfSize;
   }

   const size_type pos1 = GetXLFDNameComponentAsInteger(name, "pixel size", pos, dst.fPixelSize);
   if (dst.fPixelSize < 12)
      dst.fPixelSize = 12;
   //Real size in pixel?
   return pos1;//GetXLFDNameComponentAsInteger(name, "pixel size", pos, dst.fPixelSize);
}

//______________________________________________________________________________
size_type ParsePointSize(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignored at the moment.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "point size", pos, dummy);
}


//______________________________________________________________________________
size_type ParseHoriz(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignored at the moment.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "horizontal", pos, dummy);
}


//______________________________________________________________________________
size_type ParseVert(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignored at the moment.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "vertical", pos, dummy);
}


//______________________________________________________________________________
size_type ParseSpacing(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignored at the moment.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "spacing", pos, dummy);
}


//______________________________________________________________________________
size_type ParseAvgwidth(const std::string &name, size_type pos, XLFDName &/*dst*/)
{
   //Ignored at the moment.
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "average width", pos, dummy);
}


//______________________________________________________________________________
size_type ParseRgstry(const std::string &name, size_type pos, XLFDName &dst)
{
   return GetXLFDNameComponentAsString(name, "language", pos, dst.fRgstry);
}


//______________________________________________________________________________
size_type ParseEncoding(const std::string &name, size_type pos, XLFDName &dst)
{
   return GetXLFDNameComponentAsString(name, "encoding", pos, dst.fRgstry);
}

}//Anonymous namespace.

//______________________________________________________________________________
XLFDName::XLFDName()
            : fWeight(kFWAny),
              fSlant(kFSAny),
              fPixelSize(0)
{
}

//______________________________________________________________________________
bool ParseXLFDName(const std::string &xlfdName, XLFDName &dst)
{
   const size_type nameLength = xlfdName.length();

   assert(nameLength && "XLFD name is a string with a zero length");

   if (!nameLength) {
      ::Warning("ROOT::MacOSX::X11::ParseXLFDName: ", "XLFD name is a string with a zero length");
      return false;
   }

   try {
      if (xlfdName == "fixed" || xlfdName == "*") {
         //Is this correct XLFD name???? Who knows. Replace it.
         dst.fFamilyName = "LucidaGrande";
         dst.fPixelSize = 12;
      } else {
         size_type pos = ParseFoundry(xlfdName, 0, dst);
         if (pos + 1 < nameLength)
            pos = ParseFamilyName(xlfdName, pos, dst);

         if (pos + 1 < nameLength)
            pos = ParseWeight(xlfdName, pos, dst);
         else
            dst.fWeight = kFWMedium;

         if (pos + 1 < nameLength)
            pos = ParseSlant(xlfdName, pos, dst);
         else
            dst.fSlant = kFSRegular;

         if (pos + 1 < nameLength)
            pos = ParseSetwidth(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseAddstyle(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParsePixelSize(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParsePointSize(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseHoriz(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseVert(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseSpacing(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseAvgwidth(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseRgstry(xlfdName, pos, dst);
         if (pos + 1 < nameLength)
            pos = ParseEncoding(xlfdName, pos, dst);
      }

      return true;
   } catch (const std::exception &e) {
      ::Error("ROOT::MacOSX::Quartz::ParseXLFDName", "Failed to parse XLFD name - %s", e.what());
      return false;
   }
}

}//X11
}//MacOSX
}//ROOT
