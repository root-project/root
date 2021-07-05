/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFont
#define ROOT7_RFont

#include "ROOT/RDrawable.hxx"

#include <string>

namespace ROOT {
namespace Experimental {


/** \class RFont
\ingroup GpadROOT7
\brief Custom font configuration for the RCanvas
\author Sergey Linev <s.linev@gsi.de>
\date 2021-07-02
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFont : public RDrawable  {

   std::string fName;   ///< font name, assigned as "font-family" attribute
   std::string fSrc;    ///< font source, assigned as "src" attribute

public:

   RFont() : RDrawable("font") {}

   RFont(const std::string &name, const std::string &fname = "", const std::string &fmt = "woff2") : RFont()
   {
     SetName(name);
     SetFile(fname, fmt);
   }

   void SetName(const std::string &name) { fName = name; }
   const std::string &GetName() const { return fName; }

   void SetUrl(const std::string &url, const std::string &fmt = "woff2");
   void SetFile(const std::string &fname, const std::string &fmt = "woff2");
   void SetSrc(const std::string &src);

   const std::string &GetSrc() const { return fSrc; }
};

} // namespace Experimental
} // namespace ROOT

#endif
