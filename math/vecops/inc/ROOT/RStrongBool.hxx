// Author: Danilo Piparo CERN  06/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RStrongBool
#define ROOT_RStrongBool

#include <iostream>

namespace ROOT {
namespace VecOps {

// clang-format off
///////////////////////////////////////////////////////////////////////////////
/// \brief Strongly typed boolean type to prevent casts.
/// The storage of the value is of type int.
// clang-format on
class RStrongBool final {
public:
  RStrongBool();
  ~RStrongBool();
  template<typename T>
  RStrongBool(T val) = delete;
  RStrongBool(bool val);
  // this is inlined to workaround missing symbols errors
  explicit operator bool() const
{
	return bool(fVal);
}
private:
  int fVal;
};

} // End NS VecOps
} // End NS ROOT

// this is inlined to workaround missing symbols errors
inline std::ostream& operator<< (std::ostream& stream, ROOT::VecOps::RStrongBool sb)
{
	stream << (sb ? "true" : "false");
	return stream;
}

namespace cling {
std::string printValue(::ROOT::VecOps::RStrongBool *sb);
} // End NS Cling

#endif
