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
#include <cassert>

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
 * \param error the error. If the error is negative or zero, only the value is returned with no specific rounding
 * applied, using std::to_string
 * \param cutoff should lay between 0 and 9. If first significant digit of error starts with value <= cutoff,
 * use two significant digits instead of two for rounding. Set this value to zero to always use a
 * single digit; set this value to 9 to always use two digits.
 * \param delim delimiter between value and error printed into returned string, leave default for using ROOT's latex
 * mode.
 * \return a string with printed rounded value and error separated by "+/-" in ROOT latex mode \note The return
 * format is `A+-B` using ios::fixed with the proper precision; for very large or very small values of the error, the
 * format is changed from `A+-B` to (A'+-B')*1eX, with X being multiple of 3, respecting the corresponding precision.
 * \see https://www.bipm.org/en/doi/10.59161/jcgm100-2008e, https://physics.nist.gov/cuu/Uncertainty/
 */
std::string Round(double value, double error, unsigned int cutoff, std::string_view delim)
{
   if (error <= 0.) {
      return std::to_string(value);
   }

   int error_exponent_base10_rounded = std::floor(std::log10(error));
   const auto error_magnitude_base10 = std::pow(10., error_exponent_base10_rounded);
   const auto error_first_digit = static_cast<unsigned int>(error / error_magnitude_base10);
   assert(error_first_digit > 0 && error_first_digit < 10);
   if (error_first_digit <= cutoff) {
      const double rescaled_error = error * std::pow(10., -1. * error_exponent_base10_rounded);
      if (static_cast<unsigned int>(std::round(rescaled_error * 10) / 10) <= cutoff)
         error_exponent_base10_rounded--;
   } else if (cutoff == 0 && error_first_digit == 9) {
      const double rounded_rescaled_error = std::round(error * std::pow(10., -1. * error_exponent_base10_rounded));
      const int rounded_rescaled_error_exponent_base10_rounded = std::floor(std::log10(rounded_rescaled_error));
      const auto rounded_rescaled_error_magnitude_base10 =
         std::pow(10., rounded_rescaled_error_exponent_base10_rounded);
      const auto rounded_rescaled_error_first_digit =
         static_cast<int>(rounded_rescaled_error / rounded_rescaled_error_magnitude_base10);
      if (rounded_rescaled_error_first_digit == 1)
         error_exponent_base10_rounded++;
   }
   const int factored_out_exponent_base10 = error <= 1e-3 ? static_cast<int>(std::floor(std::log10(error) / 3)) * 3
                                                          : static_cast<int>(std::log10(error) / 3) * 3;

   std::stringstream result;
   result.setf(std::ios::fixed);
   if (error_exponent_base10_rounded - factored_out_exponent_base10 < 0) {
      result.precision(-error_exponent_base10_rounded + factored_out_exponent_base10);
   } else {
      result.precision(0);
   }
   if (factored_out_exponent_base10 != 0)
      result << "(";
   result << std::round(value * std::pow(10., -error_exponent_base10_rounded)) /
                std::pow(10., -error_exponent_base10_rounded + factored_out_exponent_base10);
   result << delim;
   result << std::round(error * std::pow(10., -error_exponent_base10_rounded)) /
                std::pow(10., -error_exponent_base10_rounded + factored_out_exponent_base10);
   if (factored_out_exponent_base10 != 0)
      result << ")*1e" << factored_out_exponent_base10;
   return result.str();
}

} // namespace ROOT
