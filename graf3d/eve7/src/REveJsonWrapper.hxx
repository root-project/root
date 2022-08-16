// @(#)root/eve7:$Id$
// Author: Jonas Hahnfeld, 2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveJsonWrapper_hxx
#define ROOT7_REveJsonWrapper_hxx

#include <nlohmann/json.hpp>

namespace ROOT {
namespace Experimental {
namespace Internal {

struct REveJsonWrapper {
  nlohmann::json &json;

  REveJsonWrapper(nlohmann::json &j) : json(j) {}

  operator nlohmann::json &() { return json; }

  // Convenience function to access properties of the wrapped object.
  template <typename Key>
  nlohmann::json::reference operator[](const Key &key) { return json.operator[](key); }
};

}
}
}

#endif
