/// \cond ROOFIT_INTERNAL

#ifndef ROOSTATS_HISTFACTORY_HISTFACTORYINTERPOLATIONCODEUTILS_H
#define ROOSTATS_HISTFACTORY_HISTFACTORYINTERPOLATIONCODEUTILS_H

#include <RooAbsReal.h>
#include <RooMsgService.h>

#include <cstddef>
#include <vector>

namespace RooStats {
namespace HistFactory {
namespace Detail {

/// Validate an interpolation code and store it in `codes[iParam]`.
///
/// Shared implementation for FlexibleInterpVar and PiecewiseInterpolation, which
/// only differ in the highest supported code. Codes outside [0, maxCode] are
/// rejected and the current code is kept. Code 3 is mapped to code 2 to preserve
/// the historical behaviour where the two were equivalent. `self` and `param` are
/// only used for the error messages.
///
/// \return true if a code was stored, so the caller can flag itself value-dirty.
inline bool setInterpolationCode(RooAbsReal const &self, const char *className, RooAbsArg const &param,
                                 std::vector<int> &codes, std::size_t iParam, int code, int maxCode)
{
   if (code < 0 || code > maxCode) {
      oocoutE(&self, InputArguments) << className << "::setInterpCode ERROR: " << param.GetName()
                                     << " with unknown interpolation code " << code << ", keeping current code "
                                     << codes[iParam] << std::endl;
      return false;
   }
   if (code == 3) {
      // In the past, code 3 was equivalent to code 2, which confused users.
      // Now, we just say that code 3 doesn't exist and default to code 2 in
      // that case for backwards compatible behavior.
      oocoutE(&self, InputArguments) << className << "::setInterpCode ERROR: " << param.GetName()
                                     << " with unknown interpolation code " << code << ", defaulting to code 2"
                                     << std::endl;
      code = 2;
   }
   codes.at(iParam) = code;
   return true;
}

} // namespace Detail
} // namespace HistFactory
} // namespace RooStats

#endif

/// \endcond
