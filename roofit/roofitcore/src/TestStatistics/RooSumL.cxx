// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <TestStatistics/RooSumL.h>
#include <RooAbsData.h>
#include <TestStatistics/RooSubsidiaryL.h>

#include <algorithm> // min, max

namespace RooFit {
namespace TestStatistics {

/** \class RooSumL
 * \ingroup Roofitcore
 *
 * \brief Likelihood class that sums over multiple -log components
 *
 * The likelihood is often a product of components, for instance when fitting simultaneous pdfs, but also when using
 * subsidiary pdfs. Hence, the negative log likelihood that we, in fact, calculate is often a sum over these components.
 * This sum is implemented by this class.
 **/

/// \param[in] pdf Raw pointer to the pdf; will not be cloned in this object.
/// \param[in] data Raw pointer to the dataset; will not be cloned in this object.
/// \param[in] components The component likelihoods.
/// \param extended Set extended term calculation on, off or use Extended::Auto to determine automatically based on the pdf whether to activate or not.
/// \warning components must be passed with std::move, otherwise it cannot be moved into the RooSumL because of the unique_ptr!
/// \note The number of events in RooSumL is that of the full dataset. Components will have their own number of events that may be more relevant.
RooSumL::RooSumL(RooAbsPdf *pdf, RooAbsData *data, std::vector<std::unique_ptr<RooAbsL>> components, RooAbsL::Extended extended)
   : RooAbsL(pdf, data,
             data->numEntries(),
             components.size(), extended), components_(std::move(components))
{}
// Developer note on the std::move() warning above:
//
// The point here was that you don't want to clone RooAbsL's too much, because they contain clones of the pdf and dataset
// that may have been mangled for optimization. You probably don't want to be doing that all the time, although it is a
// premature optimization, since we haven't timed its impact. That is the motivation behind using unique_ptrs for the
// components. The way the classes are built, the RooSumL doesn't care about what components it gets, so by definition it
// cannot create them internally, so they have to be passed in somehow. Forcing the user to call the function with a
// std::move is a way to make them fully realize that their local components will be destroyed and the contents moved
// into the RooSumL.
//
// We could change the type to an rvalue reference to make it clearer from the compiler error that std::move is
// necessary, instead of the obscure error that you get now. Compare the compiler error messages from these two types:
//
//#include <vector>
//#include <memory>
//#include <cstdio>
//
//struct Clear {
//   Clear(std::vector<std::unique_ptr<int>>&& vec) : vec_(std::move(vec)) {
//      printf("number is %d", *vec_[0]);
//   }
//
//   std::vector<std::unique_ptr<int>> vec_;
//};
//
//struct Obscure {
//   Obscure(std::vector<std::unique_ptr<int>> vec) : vec_(std::move(vec)) {
//      printf("number is %d", *vec_[0]);
//   }
//
//   std::vector<std::unique_ptr<int>> vec_;
//};
//
//int main() {
//   std::vector<std::unique_ptr<int>> vec;
//   vec.emplace_back(new int(4));
//   Clear thing(vec);
//   Obscure thingy(vec);
//}


/// \note Compared to the RooAbsTestStatistic implementation that this was taken from, we leave out Hybrid and
/// SimComponents interleaving support here. This should be implemented by a calculator (i.e. LikelihoodWrapper or
/// LikelihoodGradientWrapper derived class), if desired.
double RooSumL::evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end)
{
   // Evaluate specified range of owned GOF objects
   double ret = 0;

   // from RooAbsOptTestStatistic::combinedValue (which is virtual, so could be different for non-RooNLLVar!):
   eval_carry_ = 0;
   for (std::size_t ix = components_begin; ix < components_end; ++ix) {
      double y = components_[ix]->evaluatePartition(events, 0, 0);

      eval_carry_ += components_[ix]->getCarry();
      y -= eval_carry_;
      double t = ret + y;
      eval_carry_ = (t - ret) - y;
      ret = t;
   }

   return ret;
}

/// \note This function assumes there is only one subsidiary component.
std::tuple<double, double> RooSumL::getSubsidiaryValue()
{
   // iterate in reverse, because the subsidiary component is usually at the end:
   for (auto component = components_.rbegin(); component != components_.rend(); ++component) {
      if (dynamic_cast<RooSubsidiaryL *>((*component).get()) != nullptr) {
         double value = (*component)->evaluatePartition({0, 1}, 0, 0);
         double carry = (*component)->getCarry();
         return {value, carry};
      }
   }
   return {0, 0};
}

void RooSumL::constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) {
   for (auto& component : components_) {
      component->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
   }
}

} // namespace TestStatistics
} // namespace RooFit
