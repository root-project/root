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

#include <RooMsgService.h>
#include <RooAbsArg.h>
#include <RooAbsReal.h>

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

/// Struct to temporarily change the operation mode of a RooAbsArg until it
/// goes out of scope.
class ChangeOperModeRAII {
public:
   ChangeOperModeRAII(RooAbsArg *arg, RooAbsArg::OperMode opMode) : _arg{arg}, _oldOpMode(arg->operMode())
   {
      arg->setOperMode(opMode, /*recurse=*/false);
   }

   ChangeOperModeRAII(ChangeOperModeRAII const &other) = delete;
   ChangeOperModeRAII &operator=(ChangeOperModeRAII const &other) = delete;

   ~ChangeOperModeRAII() { _arg->setOperMode(_oldOpMode, /*recurse=*/false); }

private:
   RooAbsArg *_arg = nullptr;
   RooAbsArg::OperMode _oldOpMode;
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

namespace RooFit {
namespace Detail {

std::string makeValidVarName(std::string const &in);

void replaceAll(std::string &inOut, std::string_view what, std::string_view with);

std::string makeSliceCutString(RooArgSet const &sliceDataSet);

} // namespace Detail
} // namespace RooFit

#endif
