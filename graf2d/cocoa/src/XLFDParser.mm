#define NDEBUG

#include <stdexcept>
#include <sstream>
#include <cassert>
#include <cctype>

#include "XLFDParser.h"
#include "TError.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

namespace {

typedef std::string::size_type size_type;

//______________________________________________________________________________
template<class T>
void StringToInt(const std::string &str, const std::string &componentName, T &num)
{
   for (auto symbol : str) {
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
   //We ignore foundry at the moment(?)
   std::string dummy;
   return GetXLFDNameComponentAsString(name, "foundry", pos, dummy);
}


//______________________________________________________________________________
size_type ParseFamilyName(const std::string &name, size_type pos, XLFDName &dst)
{
   const size_type tokenEnd = GetXLFDNameComponentAsString(name, "family name", pos, dst.fFamilyName);
   
   //This is a "special case": ROOT uses it's own symbol.ttf, but
   //I can not use it to render text in a GUI, and also I can not use
   //Apple's system symbol font - it can not encode all symbols GUI wants.
   
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

   if (weight != "bold")
      dst.fWeight = FontWeight::medium;
   else
      dst.fWeight = FontWeight::bold;

   return pos;
}


//______________________________________________________________________________
size_type ParseSlant(const std::string &name, size_type pos, XLFDName &dst)
{
   //Slant can be regular or italic now.
   std::string slant;
   pos = GetXLFDNameComponentAsString(name, "slant", pos, slant);

   if (slant == "r" || slant == "R") {
      dst.fSlant = FontSlant::regular;
      return pos;
   }

   if (slant == "i" || slant == "I" || slant == "o" || slant == "O") {
      dst.fSlant = FontSlant::italic;
      return pos;
   }

   throw std::runtime_error("ParseSlant: unknown slant: " + slant);
   return pos;//never executed.
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
   return GetXLFDNameComponentAsInteger(name, "pixel size", pos, dst.fPixelSize);
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
bool ParseXLFDName(const std::string &xlfdName, XLFDName &dst)
{
   assert(xlfdName.length() && "XLFD name is a string with a zero length");

   if (!xlfdName.length()) {
      ::Warning("ROOT::MacOSX::Quartz::ParseXLFDName: ", "XLFD name is a string with a zero length");
      return false;
   }

   try {
      std::string::size_type pos = 0;

      pos = ParseFoundry(xlfdName, pos, dst);
      pos = ParseFamilyName(xlfdName, pos, dst);
      pos = ParseWeight(xlfdName, pos, dst);
      pos = ParseSlant(xlfdName, pos, dst);
      pos = ParseSetwidth(xlfdName, pos, dst);
      pos = ParseAddstyle(xlfdName, pos, dst);
      pos = ParsePixelSize(xlfdName, pos, dst);
      pos = ParsePointSize(xlfdName, pos, dst);
      pos = ParseHoriz(xlfdName, pos, dst);
      pos = ParseVert(xlfdName, pos, dst);
      pos = ParseSpacing(xlfdName, pos, dst);
      pos = ParseAvgwidth(xlfdName, pos, dst);
      pos = ParseRgstry(xlfdName, pos, dst);
      pos = ParseEncoding(xlfdName, pos, dst);

      return true;
   } catch (const std::exception &e) {
      ::Error("ROOT::MacOSX::Quartz::ParseXLFDName", "Failed to parse XLFD name - %s", e.what());
      return false;
   }
}

}//X11
}//MacOSX
}//ROOT
