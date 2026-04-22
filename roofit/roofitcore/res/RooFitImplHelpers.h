/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooFitImplHelpers_h
#define RooFit_RooFitImplHelpers_h

#include <RooAbsArg.h>
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooMsgService.h>

#include <RooNaNPacker.h>

#include <TMath.h>

#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class RooAbsPdf;
class RooAbsData;

/// Disable all caches for sub-branches in an expression tree.
/// This is helpful when an expression with cached sub-branches needs to be integrated numerically.
class DisableCachingRAII {
public:
   /// Inhibit all dirty-state propagation, and assume every node as dirty.
   /// \param[in] oldState Restore this state when going out of scope.
   DisableCachingRAII(bool oldState) : _oldState(oldState) { RooAbsArg::setDirtyInhibit(true); }

   DisableCachingRAII(DisableCachingRAII const &other) = delete;
   DisableCachingRAII &operator=(DisableCachingRAII const &other) = delete;

   ~DisableCachingRAII() { RooAbsArg::setDirtyInhibit(_oldState); }

private:
   bool _oldState;
};

/// Scope guard that temporarily changes the operation mode of one or more
/// RooAbsArg instances. Each call to change() records the arg's current
/// operMode before flipping it to the requested mode (non-recursively, i.e.
/// value clients are not touched). Destruction (or an explicit clear())
/// restores every recorded mode in LIFO order.
///
/// The class is movable but not copyable, so it can be returned from
/// functions that build up a batch of changes to hand to the caller.
class ChangeOperModeRAII {
public:
   ChangeOperModeRAII() = default;

   /// Convenience ctor: behaves like a scope guard for a single arg.
   ChangeOperModeRAII(RooAbsArg *arg, RooAbsArg::OperMode opMode) { change(arg, opMode); }

   ~ChangeOperModeRAII() { clear(); }

   ChangeOperModeRAII(ChangeOperModeRAII &&) = default;
   ChangeOperModeRAII &operator=(ChangeOperModeRAII &&) = default;
   ChangeOperModeRAII(ChangeOperModeRAII const &) = delete;
   ChangeOperModeRAII &operator=(ChangeOperModeRAII const &) = delete;

   /// Record arg's current operMode and flip it to opMode. If the current
   /// mode already equals opMode, this is a no-op (nothing to restore).
   void change(RooAbsArg *arg, RooAbsArg::OperMode opMode)
   {
      if (opMode == arg->operMode())
         return;
      _entries.emplace_back(arg, arg->operMode());
      arg->setOperMode(opMode, /*recurseADirty=*/false);
   }

   /// Restore every recorded change right away, emptying this guard.
   void clear()
   {
      for (auto it = _entries.rbegin(); it != _entries.rend(); ++it) {
         it->first->setOperMode(it->second, /*recurseADirty=*/false);
      }
      _entries.clear();
   }

   bool empty() const { return _entries.empty(); }

private:
   std::vector<std::pair<RooAbsArg *, RooAbsArg::OperMode>> _entries;
};

namespace RooHelpers {

std::pair<double, double> getRangeOrBinningInterval(RooAbsArg const *arg, const char *rangeName);

bool checkIfRangesOverlap(RooArgSet const &observables, std::vector<std::string> const &rangeNames);

std::string getColonSeparatedNameString(RooArgSet const &argSet, char delim = ':');
RooArgSet selectFromArgSet(RooArgSet const &, std::string const &names);

namespace Detail {

bool snapshotImpl(RooAbsCollection const &input, RooAbsCollection &output, bool deepCopy, RooArgSet const *observables);
RooAbsArg *cloneTreeWithSameParametersImpl(RooAbsArg const &arg, RooArgSet const *observables);

} // namespace Detail

/// Clone RooAbsArg object and reattach to original parameters.
template <class T>
std::unique_ptr<T> cloneTreeWithSameParameters(T const &arg, RooArgSet const *observables = nullptr)
{
   return std::unique_ptr<T>{static_cast<T *>(Detail::cloneTreeWithSameParametersImpl(arg, observables))};
}

std::string getRangeNameForSimComponent(std::string const &rangeName, bool splitRange, std::string const &catName);

struct BinnedLOutput {
   RooAbsPdf *binnedPdf = nullptr;
   bool isBinnedL = false;
};

BinnedLOutput getBinnedL(RooAbsPdf const &pdf);

void getSortedComputationGraph(RooAbsArg const &func, RooArgSet &out);

} // namespace RooHelpers

namespace RooFit::Detail {

std::string makeValidVarName(std::string const &in);

void replaceAll(std::string &inOut, std::string_view what, std::string_view with);

std::string makeSliceCutString(RooArgSet const &sliceDataSet);

// Inlined because this is called inside RooAbsPdf::getValV(), and therefore
// performance critical.
inline double normalizeWithNaNPacking(RooAbsPdf const &pdf, double rawVal, double normVal)
{

   if (normVal < 0. || (normVal == 0. && rawVal != 0)) {
      // Unreasonable normalisations. A zero integral can be tolerated if the function vanishes, though.
      const std::string msg = "p.d.f normalization integral is zero or negative: " + std::to_string(normVal);
      pdf.logEvalError(msg.c_str());
      return RooNaNPacker::packFloatIntoNaN(-normVal + (rawVal < 0. ? -rawVal : 0.));
   }

   if (rawVal < 0.) {
      std::stringstream ss;
      ss << "p.d.f value is less than zero (" << rawVal << "), trying to recover";
      pdf.logEvalError(ss.str().c_str());
      return RooNaNPacker::packFloatIntoNaN(-rawVal);
   }

   if (TMath::IsNaN(rawVal)) {
      pdf.logEvalError("p.d.f value is Not-a-Number");
      return rawVal;
   }

   return (rawVal == 0. && normVal == 0.) ? 0. : rawVal / normVal;
}

} // namespace RooFit::Detail

double toDouble(const char *s);
double toDouble(const std::string &s);

#endif
