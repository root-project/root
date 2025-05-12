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

#include "ROOT/StringUtils.hxx"
#include <sstream>
#include <cmath>
#include <ios>

namespace ROOT {

/// Splits a string at each character in delims.
/// The behavior mimics `str.split` from Python,
/// \param[in] str String to tokenise.
/// \param[in] delims One or more delimiters used to split the string.
/// \param[in] skipEmpty Strip empty strings from the output.
std::vector<std::string> Split(std::string_view str, std::string_view delims, bool skipEmpty /* =false */)
{
   std::vector<std::string> out;

   std::size_t beg = 0;
   std::size_t end = 0;
   while ((end = str.find_first_of(delims, beg)) != std::string::npos) {
      if (!skipEmpty || end > beg)
         out.emplace_back(str.substr(beg, end - beg));
      beg = end + 1;
   }
   if (!skipEmpty || str.size() > beg)
      out.emplace_back(str.substr(beg, str.size() - beg));

   return out;
}

/**
 * \brief Convert (round) a value and its uncertainty to string using one or two significant digits of the error
 * \param error the error. If the error is negative or zero, only the value is returned with no specific rounding applied, using std::to_string
 * \param cutoff should lay between 0 and 9. If first significant digit of error starts with value <= cutoff, use two significant digits instead of two for
 * rounding. Set this value to zero to always use a single digit; set this value to 9 to always use two digits.
 * \param delim delimiter between value and error printed into returned string, leave default for using ROOT's latex mode
 * \return a string with printed rounded value and error separated by "+/-" in ROOT latex mode
 * \note The return format is `A+-B` using ios::fixed with the proper precision;
 * for very large or very small values of the error, the format is changed from `A+-B` to (A'+-B')*1eX, with X being multiple of 3, respecting the corresponding precision.
 * \see https://www.bipm.org/en/doi/10.59161/jcgm100-2008e, https://physics.nist.gov/cuu/Uncertainty/
 */
std::string Round(const double value, const double error, const unsigned short cutoff, const std::string delim)
{
   if(error <= 0.)
   {
      return std::to_string(value);
   }

   int nexp10 = std::floor(std::log10(error));
   const auto scale = std::pow(10., nexp10);
   const auto first_digit = static_cast<int>(error / scale);
   assert (first_digit > 0 && first_digit < 10);
   if (first_digit <= cutoff) {
      const double rerror = error * std::pow(10., -1. * nexp10);
      if (static_cast<int>(std::round(rerror * 10) / 10) <= cutoff)
         nexp10--;
   } else if (cutoff == 0 && first_digit == 9) {
      const double rerror = std::round(error * std::pow(10., -1. * nexp10));
      const int rnexp10 = std::floor(std::log10(rerror));
      const auto rscale = std::pow(10., rnexp10);
      const auto rfirst_digit = static_cast<int>(rerror / rscale);
      if (rfirst_digit == 1)
         nexp10++;
   }

   std::stringstream sv, se;
   sv.setf(std::ios::fixed);
   se.setf(std::ios::fixed);
   const int maxExpo = error <= 1e-3 ? static_cast<int>(std::floor(std::log10(error) / 3)) * 3 : static_cast<int>(std::log10(error) / 3) * 3;
   if(nexp10 - maxExpo < 0) {
      sv.precision(-nexp10 + maxExpo);
      se.precision(-nexp10 + maxExpo);
   }
   else {
      sv.precision(0);
      se.precision(0);
   }
   sv << std::round(value * std::pow(10., -nexp10))/std::pow(10., -nexp10 + maxExpo);
   se << std::round(error * std::pow(10., -nexp10))/std::pow(10., -nexp10 + maxExpo);

   return (maxExpo != 0 ? "(" : "") + sv.str() + delim + se.str() + (maxExpo != 0 ? ")*1e" + std::to_string(maxExpo) : "") ;
}

} // namespace ROOT
