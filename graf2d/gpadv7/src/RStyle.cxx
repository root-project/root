/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RStyle.hxx>

#include <ROOT/RDrawable.hxx>
#include <ROOT/RLogger.hxx>


// #include "RStyleReader.hxx" // in src/


using namespace std::string_literals;


///////////////////////////////////////////////////////////////////////////////
/// Evaluate style

const ROOT::Experimental::RAttrMap::Value_t *ROOT::Experimental::RStyle::Eval(const std::string &field, const RDrawable &drawable) const
{
   for (const auto &block : fBlocks) {
      if (drawable.MatchSelector(block.selector)) {
         auto res = block.map.Find(field);
         if (res)
            return res;
      }
   }

   return nullptr;
}


///////////////////////////////////////////////////////////////////////////////
/// Parse string with CSS code inside

void ROOT::Experimental::RStyle::Clear()
{
   fBlocks.clear();
}

///////////////////////////////////////////////////////////////////////////////
/// Parse string with CSS code inside
/// All data will be append to existing style records

bool ROOT::Experimental::RStyle::ParseString(const std::string &css_code)
{
   if (css_code.empty())
      return true;

   int len = css_code.length(), pos = 0;

   auto skip_empty = [&css_code, &pos, len] () -> bool {
      bool skip_until_newline = false, skip_until_endblock = false;

      while (pos < len) {
         if (css_code[pos] == '\n') {
            skip_until_newline = false;
            ++pos;
            continue;
         }

         if (skip_until_endblock && (css_code[pos] == '*') && (pos+1 < len) && (css_code[pos+1] == '/')) {
            pos+=2;
            skip_until_endblock = false;
            continue;
         }

         if (skip_until_newline || (css_code[pos] == ' ') || (css_code[pos] == '\t')) {
            ++pos;
            continue;
         }

         if ((css_code[pos] == '/') && (pos+1 < len)) {
            if (css_code[pos+1] == '/') {
               pos+=2;
               skip_until_newline = true;
               continue;
            } else if (css_code[pos+1] == '*')  {
               pos+=2;
               skip_until_endblock = true;
               continue;
            }
         }

         return true;
      }

      return false;
   };

   auto check_symbol = [] (char symbol, bool isfirst = false) -> bool {
      if (((symbol >= 'a') && (symbol <= 'z')) ||
          ((symbol >= 'A') && (symbol <= 'Z')) || (symbol == '_')) return true;
      return (!isfirst && (symbol>='0') && (symbol<='9'));
   };

   auto scan_identifier = [&check_symbol, &css_code, &pos, len] (bool selector = false) -> std::string {
      if (pos >= len) return ""s;

      int pos0 = pos;

      // start symbols of selector
      if (selector && ((css_code[pos] == '.') || (css_code[pos] == '#'))) ++pos;

      while ((pos < len) && check_symbol(css_code[pos])) ++pos;

      return css_code.substr(pos0, pos-pos0);
   };

   auto scan_value = [&css_code, &pos, len] () -> std::string {
       if (pos >= len) return ""s;

       int pos0 = pos;

       while ((pos < len) && (css_code[pos] != ';')) {
          if (css_code[pos] == '\n') return ""s;
          pos++;
       }
       if (pos >= len) return ""s;
       ++pos;

       return css_code.substr(pos0, pos - pos0 - 1);
   };

   RStyle newstyle;

   while (pos < len) {

      if (!skip_empty())
         return false;

      auto sel = scan_identifier(true);
      if (sel.empty()) {
         R__ERROR_HERE("rstyle") << "Fail to find selector";
         return false;
      }

      if (!skip_empty())
         return false;

      if (css_code[pos] != '{') {
         R__ERROR_HERE("rstyle") << "Fail to find starting {";
         return false;
      }

      ++pos;

      if (!skip_empty())
         return false;

      auto &map = newstyle.AddBlock(sel);

      while (css_code[pos] != '}') {
         auto name = scan_identifier();
         if (name.empty()) {
            R__ERROR_HERE("rstyle") << "not able to extract identifier";
            return false;
         }

         if (!skip_empty())
            return false;

         if (css_code[pos] != ':') {
            R__ERROR_HERE("rstyle") << "not able to find separator :";
            return false;
         }

         ++pos;
         if (!skip_empty())
            return false;

         auto value = scan_value();
         if (value.empty()) {
            R__ERROR_HERE("rstyle") << "not able to find value";
            return false;
         }

         map.AddString(name, value);

         if (!skip_empty())
            return false;
      }
      ++pos;

      skip_empty(); // after closing } end of file is possible
   }

   // finally move all read blocks to this
   fBlocks.splice(fBlocks.end(), newstyle.fBlocks);

   return true;
}
