/// \file RPadLength.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-07-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RPadLength.hxx"

#include <ROOT/TLogger.hxx>

#include <algorithm> // std::transform
#include <cctype> // std::tolower

namespace {
   enum ELengthKind {
      kNormal,
      kPixel,
      kUser
   };

   struct RLengthElement {
      ELengthKind fKind;
      double fVal = 0.;
   };

   struct RLengthParseElements {
      char fSignOrOp = '+';
      std::string fNumber;
      std::string fUnit;
      int fIndex = 0;

      bool Done() const { return fIndex == 3; }

      std::string SetMaybeSignOrOpAndGetError(std::string_view tok)
      {
         fIndex = 1;
         if (tok.front() == '-' || tok.front() == '+')
            fSignOrOp = tok.front();

         if (tok.length() > 1)
            return SetNumberAndGetError(tok.substr(1));

         return {};
      }

      std::string SetNumberAndGetError(std::string_view tok)
      {
         // must be all numbers, up to possible remaining unit.
         fIndex = 2;

         if (tok.front() == '-') {
            if (fSignOrOp == '-')
               fSignOrOp = '+';
            else
               fSignOrOp = '-';
            tok = tok.substr(1);
         } else if (tok.front() == '+')
            tok = tok.substr(1);

         bool hadDot = false;
         for (const char &c: tok) {
            if (std::isdigit(c) || c == '.') {
               if (c == '.') {
                  if (hadDot) {
                     std::string err = "syntax error: multiple '.' in ";
                     err += std::string(tok);
                     return err;
                  }
                  hadDot = true;
               }

               fNumber += c;
            } else {
               return SetUnitAndGetError(tok.substr(&c - tok.data()));
            }
         }
         return {};
      }

      std::string SetUnitAndGetError(std::string_view tok)
      {
         fIndex = 3;
         fUnit = std::string(tok);
         std::transform(fUnit.begin(), fUnit.end(), fUnit.begin(), [](char c) { return std::tolower(c); });
         return "";
      }

      std::string SetNextAndGetError(std::string_view tok)
      {
         switch (fIndex) {
            case 0: return SetMaybeSignOrOpAndGetError(tok);
            case 1: return SetNumberAndGetError(tok);
            case 2: return SetUnitAndGetError(tok);
            default: return "invalid syntax: too many / duplicate elements";
         }
         return {};
      }

      std::pair<RLengthElement, std::string> ToLengthElement() const {
         RLengthElement ret;
         std::size_t posStrToD;
         if (fNumber.empty()) {
            return {{}, "empty floating point number"};
         }
         ret.fVal = std::stod(fNumber, &posStrToD);
         if (posStrToD != fNumber.length())
            return {{}, std::string("invalid floating point number ") + fNumber};
         
         if (fSignOrOp == '-')
            ret.fVal *= -1;

         if (fUnit == "normal")
            ret.fKind = kNormal;
         else if (fUnit == "px" || fUnit == "pixel")
            ret.fKind = kPixel;
         else if (fUnit == "user")
            ret.fKind = kUser;
         else
            return {{}, std::string("invalid unit, expect normal, px, or user but got ") + fUnit};
         return {ret, ""};
      }
   };

   static std::string HandleToken(std::string &tok, RLengthParseElements &parse, ROOT::Experimental::RPadLength &obj) {
      std::string err = parse.SetNextAndGetError(tok);
      tok.clear();
      if (!err.empty())
         return err;

      if (parse.Done()) {
         std::pair<RLengthElement, std::string> parseRes = parse.ToLengthElement();
         if (!parseRes.second.empty())
            return parseRes.second;
         switch (parseRes.first.fKind) {
            case kNormal: obj += ROOT::Experimental::RPadLength::Normal(parseRes.first.fVal); break;
            case kPixel: obj += ROOT::Experimental::RPadLength::Pixel(parseRes.first.fVal); break;
            case kUser: obj += ROOT::Experimental::RPadLength::User(parseRes.first.fVal); break;
         }

         // Restart.
         parse = {};
      }
      return {};
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize a RPadLength from a style string.
/// Syntax: a series of numbers separated by "+", where each number is
/// followed by one of `px`, `user`, `normal` to specify an extent in pixel,
/// user or normal coordinates. Spaces between any part is allowed.
/// Example: `100 px + 0.1 user, 0.5 normal` is a `RPadExtent{100_px + 0.1_user, 0.5_normal}`.

void ROOT::Experimental::RPadLength::SetFromAttrString(const std::string &name, const std::string &attrStrVal)
{
   *this = {}; // Value-initialize this.
   std::string tok;
   RLengthParseElements parse;
   
   for (const char c: attrStrVal) {
      if (c == ' ') {
         if (!tok.empty()) {
            std::string err = HandleToken(tok, parse, *this);
            if (!err.empty()) {
               R__ERROR_HERE("Gpad") << "Invalid syntax in '" << attrStrVal
                  << "' while parsing pad length for " << name << ": " << err;
               return;
            }
         }
      } else
         tok += c;
   }

   // Handle last token:
   if (!tok.empty()) {
      std::string err = HandleToken(tok, parse, *this);
      if (!err.empty()) {
         R__ERROR_HERE("Gpad") << "Invalid syntax in '" << attrStrVal
            << "' while parsing pad length for " << name << ": " << err;
         return;
      }
   }
   if (parse.fIndex != 0) {
      R__ERROR_HERE("Gpad") << "Invalid syntax in '" << attrStrVal
         << "' while parsing pad length for " << name
         << ": missing elements, expect [+-] number (normal|px|user)";
      return;
   }
}

std::string ROOT::Experimental::RPadLength::PadLengthToString(const RPadLength& len)
{
   std::stringstream strm;
   strm << len.fNormal << " normal + " << len.fPixel << " px + " << len.fUser << " user";
   return strm.str();
}