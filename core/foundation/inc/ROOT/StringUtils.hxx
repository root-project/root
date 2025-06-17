/// \file ROOT/StringUtils.hxx
/// \ingroup Base StdExt
/// \author Jonas Rembser <jonas.rembser@cern.ch>
/// \date 2021-08-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_StringUtils
#define ROOT_StringUtils

#include <string_view>

#include <string>
#include <vector>
#include <numeric>
#include <iterator>

namespace ROOT {

std::vector<std::string> Split(std::string_view str, std::string_view delims, bool skipEmpty = false);

/**
 * \brief Concatenate a list of strings with a separator
 * \tparam StringCollection_t Any container of strings (vector, initializer_list, ...)
 * \param[in] sep Separator inbetween the strings.
 * \param[in] strings container of strings
 * \return the sep-delimited concatenation of strings
 */
template <class StringCollection_t>
std::string Join(const std::string &sep, StringCollection_t &&strings)
{
   if (strings.empty())
      return "";
   return std::accumulate(std::next(std::begin(strings)), std::end(strings), strings[0],
                          [&sep](auto const &a, auto const &b) { return a + sep + b; });
}

std::string Round(double value, double error, unsigned int cutoff = 1, std::string_view delim = "#pm");

} // namespace ROOT

#endif
