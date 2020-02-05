/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RStyleReader.hxx" // in src/

#include <ROOT/RColor.hxx>
#include <ROOT/RLogger.hxx>

#include <TROOT.h>
#include <TSystem.h>

#include <fstream>
#include <string>
#include <cctype>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

/*

void RStyleReader::ReadDefaults()
{
   RStyleReader reader(fAttrs);
   reader.AddFromStyleFile(std::string(TROOT::GetEtcDir().Data()) + "/system.rootstylerc");
   reader.AddFromStyleFile(gSystem->GetHomeDirectory() + "/.rootstylerc");
   reader.AddFromStyleFile(".rootstylerc");
}

namespace {
   /// Possibly remove trailing comment starting with '#'
   void RemoveComment(std::string& line)
   {
      auto posComment = line.find('#');
      if (posComment != std::string::npos)
         line.erase(posComment - 1);
   }

   /// Create a view on the key and one on the value of the line.
   std::array<std::string_view, 2> SeparateKeyValue(const std::string& line)
   {
      std::string::size_type posColon = line.find(':');
      if (posColon == std::string::npos)
         return {};
      std::array<std::string_view, 2> ret{{line, line}};
      ret[0] = ret[0].substr(0, posColon);
      ret[1] = ret[1].substr(posColon + 1);
      return ret;
   }

   /// Remove leading an trailing whitespace from string_view.
   std::string_view TrimWhitespace(std::string_view view)
   {
      std::string_view ret(view);
      auto begin = ret.begin();
      for (auto end = ret.end(); begin != end; ++begin) {
         if (!std::isspace(*begin))
            break;
      }
      ret.remove_prefix(begin - ret.begin());

      auto rbegin = ret.rbegin();
      for (auto rend = ret.rend(); rbegin != rend; ++rbegin) {
         if (!std::isspace(*rbegin))
            break;
      }
      ret.remove_suffix(rbegin - ret.rbegin());
      return ret;
   }
}

bool RStyleReader::AddFromStyleFile(const std::string &filename)
{
   std::ifstream in(filename);
   if (!in)
      return false;
   std::string styleName;
   std::string line;
   long long lineNo = 0;
   while (std::getline(in, line)) {
      ++lineNo;
      RemoveComment(line);
      auto noSpaceLine = TrimWhitespace(line);
      if (noSpaceLine.compare(0, 1, "[") == 0 && noSpaceLine.compare(noSpaceLine.length() - 1, 1, "]")) {
         // A section introducing a style name.
         noSpaceLine.remove_prefix(1); // [
         noSpaceLine.remove_suffix(1); // ]
         noSpaceLine = TrimWhitespace(noSpaceLine);
         styleName = std::string(noSpaceLine);
         continue;
      }

      auto keyValue = SeparateKeyValue(line);
      for (auto& kv: keyValue)
         kv = TrimWhitespace(kv);
      if (keyValue[0].empty()) {
         R__ERROR_HERE("GPrimitive") << "Syntax error in style file " << filename << ":" << lineNo << ": "
            << "missing key in line \n" << line << "\nInoring line.";
         continue;
      }

      fAttrs[styleName].GetAttribute(std::string(keyValue[0])) = std::string(keyValue[1]);
   }
   return true;
}
*/
