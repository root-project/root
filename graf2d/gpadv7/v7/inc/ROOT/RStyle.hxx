/// \file ROOT/RStyle.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-10-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RStyle
#define ROOT7_RStyle

#include <ROOT/RDrawingAttr.hxx>

#include <ROOT/RStringView.hxx>

#include <string>
#include <tuple>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RStyle
  A set of defaults for graphics attributes, e.g. for histogram fill color, line width, frame offsets etc.
  */

class RStyle {
public:

   struct Block_t {
      std::string selector;
      RAttrValues::Map_t map; ///<    container
      Block_t() = default;
      Block_t(const std::string &_selector) : selector(_selector) {}

      Block_t(const Block_t &) {} // dummy, should not be used, but appears in dictionary
      Block_t& operator=(const Block_t &) = delete;
   };

   const RAttrValues::Value_t *Eval(const std::string &type, const std::string &user_class, const std::string &field) const;

   RAttrValues::Map_t &AddBlock(const std::string &selector)
   {
      fBlocks.emplace_back(selector);
      return fBlocks.back().map;
   }

private:
   std::list<Block_t> fBlocks;  // use std::list to avoid calling of Block_t copy constructor
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RStyle
