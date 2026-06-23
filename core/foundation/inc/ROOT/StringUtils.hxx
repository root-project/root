/// \file ROOT/StringUtils.hxx
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
#include <utility>

namespace ROOT {

std::vector<std::string> Split(std::string_view str, std::string_view delims, bool skipEmpty = false);

/// Given a string `str`, returns a pair of string views into it: the first containing the substring preceding the
/// first instance of `splitter`  and the second containing the substring following it. Note that:
///   1. the first instance of `splitter` will not appear in either string;
///   2. if `splitter` does not appear, or appears as the last character, the second string view will be empty;
///   3. if `splitter` appears as the first character, the first string view will be empty;
///   4. if `str` is empty so will both views be.
/// IMPORTANT: The lifetime of the returned string views is the same as `str`.
std::pair<std::string_view, std::string_view> SplitAt(std::string_view str, char splitter);

/**
 * \brief Concatenate a list of strings with a separator
 * \tparam StringCollection_t Any container of strings (vector, initializer_list, ...)
 * \param[in] sep Separator inbetween the strings.
 * \param[in] begin Beginning of the strings container
 * \param[in] end Past-the-end iterator of the strings container
 * \return the sep-delimited concatenation of strings
 */
template <typename InputIt_t>
std::string Join(const std::string &sep, InputIt_t begin, InputIt_t end)
{
   if (begin == end)
      return "";

   return std::accumulate(std::next(begin), end, *begin, [&sep](auto const &a, auto const &b) { return a + sep + b; });
}

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
   return Join(sep, std::begin(strings), std::end(strings));
}

std::string Round(double value, double error, unsigned int cutoff = 1, std::string_view delim = "#pm");

inline bool StartsWith(std::string_view string, std::string_view prefix)
{
   return string.size() >= prefix.size() && string.substr(0, prefix.size()) == prefix;
}

inline bool EndsWith(std::string_view string, std::string_view suffix)
{
   return string.size() >= suffix.size() && string.substr(string.size() - suffix.size(), suffix.size()) == suffix;
}

} // namespace ROOT

#endif
