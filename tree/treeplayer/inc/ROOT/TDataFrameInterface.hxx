// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDATAFRAME_INTERFACE
#define ROOT_TDATAFRAME_INTERFACE

#include "ROOT/TActionResultProxy.hxx"
#include "ROOT/TDFNodes.hxx"
#include "ROOT/TDFOperations.hxx"
#include "ROOT/TDFUtils.hxx"
#include "TClass.h"
#include "TH1.h" // For Histo actions
#include "TH2.h" // For Histo actions
#include "TH3.h" // For Histo actions
#include "TInterpreter.h"
#include "TProfile.h"   // For Histo actions
#include "TProfile2D.h" // For Histo actions
#include "TRegexp.h"
#include "TROOT.h" // IsImplicitMTEnabled

#include <initializer_list>
#include <memory>
#include <string>
#include <typeinfo>
#include <type_traits> // is_same, enable_if

namespace ROOT {

namespace Detail {
struct TDataFrameGuessedType {
};
}

namespace Internal {

template <typename TDFNode, typename ActionType, typename BranchType, typename ActionResultType>
ROOT::Experimental::TActionResultProxy<ActionResultType> CallCreateAction(TDFNode *node, const BranchNames_t &bl,
                                                                          const std::shared_ptr<ActionResultType> &r,
                                                                          BranchType *)
{
   return node->template CreateAction<ActionType, BranchType, ActionResultType>(bl, r, nullptr);
}
} // namespace Internal

namespace Experimental {

// forward declarations
class TDataFrame;

/**
* \class ROOT::Experimental::TDataFrameInterface
* \ingroup dataframe
* \brief The public interface to the TDataFrame federation of classes: TDataFrameImpl, TDataFrameFilter,
* TDataFrameBranch
* \tparam T One of the TDataFrameImpl, TDataFrameFilter, TDataFrameBranch classes. The user never specifies this type
* manually.
*/
template <typename Proxied>
class TDataFrameInterface {
   friend std::string cling::printValue(ROOT::Experimental::TDataFrame *tdf); // For a nice printing at the prompt
   template <typename T>
   friend class TDataFrameInterface;
   template <typename TDFNode, typename ActionType, typename BranchType, typename ActionResultType>
   friend TActionResultProxy<ActionResultType> ROOT::Internal::CallCreateAction(
      TDFNode *, const BranchNames_t &, const std::shared_ptr<ActionResultType> &, BranchType *);

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] bn Names of the branches in input to the filter function.
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// Append a filter node at the point of the call graph corresponding to the
   /// object this method is called on.
   /// The callable `f` should not have side-effects (e.g. modification of an
   /// external or static variable) to ensure correct results when implicit
   /// multi-threading is active.
   ///
   /// TDataFrame only evaluates filters when necessary: if multiple filters
   /// are chained one after another, they are executed in order and the first
   /// one returning false causes the event to be discarded.
   /// Even if multiple actions or transformations depend on the same filter,
   /// it is executed once per entry. If its result is requested more than
   /// once, the cached result is served.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase> Filter(F f, const BranchNames_t &bn = {},
                                                                  const std::string &name = "")
   {
      ROOT::Internal::CheckFilter(f);
      auto                 df       = GetDataFrameChecked();
      const BranchNames_t &defBl    = df->GetDefaultBranches();
      auto                 nArgs    = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t::fgSize;
      const BranchNames_t &actualBl = ROOT::Internal::PickBranchNames(nArgs, bn, defBl);
      using DFF_t                   = ROOT::Detail::TDataFrameFilter<F, Proxied>;
      auto FilterPtr                = std::make_shared<DFF_t>(std::move(f), actualBl, *fProxiedPtr, name);
      fProxiedPtr->IncrChildrenCount();
      df->Book(FilterPtr);
      TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase> tdf_f(FilterPtr, fImplWeakPtr);
      return tdf_f;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase> Filter(F f, const std::string &name)
   {
      // The sfinae is there in order to pick up the overloaded method which accepts two strings
      // rather than this template method.
      return Filter(f, {}, name);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] bn Names of the branches in input to the filter function.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F>
   TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase> Filter(F f, const std::initializer_list<std::string> &bn)
   {
      return Filter(f, BranchNames_t{bn});
   }


   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] expression The filter expression in C++
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// The expression is just in time compiled and used to filter entries. The
   /// variable names to be used inside are the names of the branches. Only
   /// valid C++ is accepted.
   /// Refer to the first overload of this method for the full documentation.
   TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>
   Filter(const std::string &expression, const std::string& name = "")
   {

      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      auto branches = tree->GetListOfBranches();

      // Check what branches and temporary branches are used in the expression
      // To help matching the regex
      std::string paddedExpr = " " + expression + " ";
      int paddedExprLen = paddedExpr.size();
      static const std::string regexBit("[^a-zA-Z0-9_]");
      std::vector<std::string> usedBranches;
      for (auto bro : *branches) {
         auto brName = bro->GetName();
         std::string bNameRegexContent = regexBit + brName + regexBit;
         TRegexp bNameRegex(bNameRegexContent.c_str());
         if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
            usedBranches.emplace_back(brName);
         }
      }
      for (auto brName : df->GetTmpBranches()) {
         std::string bNameRegexContent = regexBit + brName + regexBit;
         TRegexp bNameRegex(bNameRegexContent.c_str());
         if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
            usedBranches.emplace_back(brName.c_str());
         }
      }

      auto exprNeedsVariables = !usedBranches.empty();

      // Move to the preparation of the jitting
      // We put all of the jitted entities in a namespace called
      // __tdf_filter_N, where N is a monotonically increasing index.
      TInterpreter::EErrorCode interpErrCode;
      std::vector<std::string> usedBranchesTypes;
      std::stringstream ss;
      static unsigned int iFilter = 0U;
      ss << "__tdf_filter_" << typeid(this).name() << "_" << iFilter++;
      const auto nsName = ss.str();
      ss.str("");

      if (exprNeedsVariables) {
         // Declare a namespace and inside it the variables in the cut expression
         ss << "namespace " << nsName;
         ss << " {\n";
         for (auto brName : usedBranches) {
            auto brTypeName =
               ROOT::Internal::ColumnName2ColumnTypeName(brName, *df->GetTree(), df->GetBookedBranch(brName));
            ss << brTypeName << " " << brName << ";\n";
            usedBranchesTypes.emplace_back(brTypeName);
         }
         ss << "}";
         auto variableDeclarations = ss.str();
         ss.str("");
         // We need ProcessLine to trigger auto{parsing,loading} where needed
         gInterpreter->ProcessLine(variableDeclarations.c_str(), &interpErrCode);
         if (TInterpreter::EErrorCode::kNoError != interpErrCode) {
            std::string msg = "Cannot declare these variables ";
            msg += " ";
            msg += variableDeclarations;
            if (TInterpreter::EErrorCode::kNoError != interpErrCode) {
               msg += "\nInterpreter error code is " + std::to_string(interpErrCode) + ".";
            }
            throw std::runtime_error(msg);
         }
      }

      // Declare within the same namespace, the expression to make sure it
      // is proper C++
      ss << "namespace "<< nsName << "{ auto res = " << expression << ";}\n";
      // Headers must have been parsed and libraries loaded: we can use Declare
      if (!gInterpreter->Declare(ss.str().c_str())) {
         std::string msg = "Cannot interpret this expression: ";
         msg += " ";
         msg += ss.str();
         throw std::runtime_error(msg);
      }


      // Now we build the lambda and we invoke Filter with it in the jitted world
      ss.str("");
      ss << "[](";
      for (unsigned int i=0 ; i<usedBranchesTypes.size(); ++i) {
         // We pass by reference to avoid expensive copies
         ss << usedBranchesTypes[i] << "& " << usedBranches[i] << ", ";
      }
      if (!usedBranchesTypes.empty()) ss.seekp(-2,ss.cur);
      ss << "){ return " << expression << ";}";
      auto filterLambda = ss.str();

      ss.str("");
      ss << "((" << GetNodeTypeName() << "*)" << this << ")->Filter(" << filterLambda << ", {";
      for (auto brName : usedBranches) {
         ss << "\"" << brName << "\", ";
      }
      if (exprNeedsVariables) ss.seekp(-2,ss.cur); // remove the last ",
      ss << "}, \"" << name << "\");";

      auto retVal = gInterpreter->ProcessLine(ss.str().c_str(), &interpErrCode);
      if (TInterpreter::EErrorCode::kNoError != interpErrCode || !retVal) {
         std::string msg = "Cannot interpret the invocation to Filter: ";
            msg += " ";
            msg += ss.str();
            if (TInterpreter::EErrorCode::kNoError != interpErrCode) {
               msg += "\nInterpreter error code is " + std::to_string(interpErrCode) + ".";
            }
            throw std::runtime_error(msg);
      }

      return *(TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>*) retVal;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a temporary branch
   /// \param[in] name The name of the temporary branch.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the
   /// temporary value. Returns the value that will be assigned to the temporary branch.
   /// \param[in] bl Names of the branches in input to the producer function.
   ///
   /// Create a temporary branch that will be visible from all subsequent nodes
   /// of the functional chain. The `expression` is only evaluated for entries that pass
   /// all the preceding filters.
   /// A new variable is created called `name`, accessible as if it was contained
   /// in the dataset from subsequent transformations/actions.
   ///
   /// Use cases include:
   ///
   /// * caching the results of complex calculations for easy and efficient multiple access
   /// * extraction of quantities of interest from complex objects
   /// * branch aliasing, i.e. changing the name of a branch
   ///
   /// An exception is thrown if the name of the new branch is already in use
   /// for another branch in the TTree.
   template <typename F>
   TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase> AddColumn(const std::string &name, F expression,
                                                                     const BranchNames_t &bl = {})
   {
      auto df = GetDataFrameChecked();
      ROOT::Internal::CheckTmpBranch(name, df->GetTree());
      const BranchNames_t &defBl    = df->GetDefaultBranches();
      auto                 nArgs    = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t::fgSize;
      const BranchNames_t &actualBl = ROOT::Internal::PickBranchNames(nArgs, bl, defBl);
      using DFB_t                   = ROOT::Detail::TDataFrameBranch<F, Proxied>;
      auto BranchPtr                = std::make_shared<DFB_t>(name, std::move(expression), actualBl, *fProxiedPtr);
      fProxiedPtr->IncrChildrenCount();
      df->Book(BranchPtr);
      TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase> tdf_b(BranchPtr, fImplWeakPtr);
      return tdf_b;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range
   /// \param[in] start How many entries to discard before resuming processing.
   /// \param[in] stop Total number of entries that will be processed before stopping. 0 means "never stop".
   /// \param[in] stride Process one entry every `stride` entries. Must be strictly greater than 0.
   ///
   /// Ranges are only available if EnableImplicitMT has _not_ been called. Multi-thread ranges are not supported.
   TDataFrameInterface<ROOT::Detail::TDataFrameRangeBase> Range(unsigned int start, unsigned int stop,
                                                                unsigned int stride = 1)
   {
      // check invariants
      if (stride == 0 || (stop != 0 && stop < start))
         throw std::runtime_error("Range: stride must be strictly greater than 0 and stop must be greater than start.");
      if (ROOT::IsImplicitMTEnabled())
         throw std::runtime_error("Range was called with ImplicitMT enabled. Multi-thread ranges are not supported.");

      auto df       = GetDataFrameChecked();
      using Range_t = ROOT::Detail::TDataFrameRange<Proxied>;
      auto RangePtr = std::make_shared<Range_t>(start, stop, stride, *fProxiedPtr);
      fProxiedPtr->IncrChildrenCount();
      df->Book(RangePtr);
      TDataFrameInterface<ROOT::Detail::TDataFrameRangeBase> tdf_r(RangePtr, fImplWeakPtr);
      return tdf_r;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range
   /// \param[in] stop Total number of entries that will be processed before stopping. 0 means "never stop".
   ///
   /// See the other Range overload for a detailed description.
   TDataFrameInterface<ROOT::Detail::TDataFrameRangeBase> Range(unsigned int stop)
   {
      return Range(0, stop, 1);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined
   /// calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// The callable `f` is invoked once per entry. This is an *instant action*:
   /// upon invocation, an event loop as well as execution of all scheduled actions
   /// is triggered.
   /// Users are responsible for the thread-safety of this callable when executing
   /// with implicit multi-threading enabled (i.e. ROOT::EnableImplicitMT).
   template <typename F>
   void Foreach(F f, const BranchNames_t &bl = {})
   {
      namespace IU = ROOT::Internal::TDFTraitsUtils;
      using Args_t = typename IU::TFunctionTraits<decltype(f)>::ArgsNoDecay_t;
      using Ret_t  = typename IU::TFunctionTraits<decltype(f)>::Ret_t;
      ForeachSlot(IU::AddSlotParameter<Ret_t>(f, Args_t()), bl);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function requiring a processing slot index on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined
   /// calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// Same as `Foreach`, but the user-defined function takes an extra
   /// `unsigned int` as its first parameter, the *processing slot index*.
   /// This *slot index* will be assigned a different value, `0` to `poolSize - 1`,
   /// for each thread of execution.
   /// This is meant as a helper in writing thread-safe `Foreach`
   /// actions when using `TDataFrame` after `ROOT::EnableImplicitMT()`.
   /// The user-defined processing callable is able to follow different
   /// *streams of processing* indexed by the first parameter.
   /// `ForeachSlot` works just as well with single-thread execution: in that
   /// case `slot` will always be `0`.
   template <typename F>
   void ForeachSlot(F f, const BranchNames_t &bl = {})
   {
      auto                 df       = GetDataFrameChecked();
      const BranchNames_t &defBl    = df->GetDefaultBranches();
      auto                 nArgs    = ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t::fgSize;
      const BranchNames_t &actualBl = ROOT::Internal::PickBranchNames(nArgs - 1, bl, defBl);
      using Op_t                    = ROOT::Internal::Operations::ForeachSlotOperation<F>;
      using DFA_t                   = ROOT::Internal::TDataFrameAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(std::move(f)), actualBl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      df->Run();
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a branch
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the branch to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] branchName The branch to be reduced. If omitted, the default branch is used instead.
   ///
   /// A reduction takes two values of a branch and merges them into one (e.g.
   /// by summing them, taking the maximum, etc). This action performs the
   /// specified reduction operation on all branch values, returning
   /// a single value of the same type. The callable f must satisfy the general
   /// requirements of a *processing function* besides having signature `T(T,T)`
   /// where `T` is the type of branch.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultPtr documentation.
   template <typename F, typename T = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Ret_t>
   TActionResultProxy<T> Reduce(F f, const std::string &branchName = {})
   {
      static_assert(std::is_default_constructible<T>::value,
                    "reduce object cannot be default-constructed. Please provide an initialisation value (initValue)");
      return Reduce(std::move(f), branchName, T());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a branch
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the branch to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] branchName The branch to be reduced. If omitted, the default branch is used instead.
   /// \param[in] initValue The reduced object is initialised to this value rather than being default-constructed
   ///
   /// See the description of the other Reduce overload for more information.
   template <typename F, typename T = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Ret_t>
   TActionResultProxy<T> Reduce(F f, const std::string &branchName, const T &initValue)
   {
      using Args_t = typename ROOT::Internal::TDFTraitsUtils::TFunctionTraits<F>::Args_t;
      ROOT::Internal::CheckReduce(f, Args_t());
      auto         df        = GetDataFrameChecked();
      unsigned int nSlots    = df->GetNSlots();
      auto         bl        = GetBranchNames<T>({branchName}, "reduce branch values");
      auto         redObjPtr = std::make_shared<T>(initValue);
      using Op_t             = ROOT::Internal::Operations::ReduceOperation<F, T>;
      using DFA_t            = typename ROOT::Internal::TDataFrameAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(std::move(f), redObjPtr, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(redObjPtr, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the number of entries processed (*lazy action*)
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   TActionResultProxy<unsigned int> Count()
   {
      auto         df     = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto         cSPtr  = std::make_shared<unsigned int>(0);
      using Op_t          = ROOT::Internal::Operations::CountOperation;
      using DFA_t         = ROOT::Internal::TDataFrameAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(cSPtr, nSlots), BranchNames_t({}), *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(cSPtr, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a collection of values of a branch (*lazy action*)
   /// \tparam T The type of the branch.
   /// \tparam COLL The type of collection used to store the values.
   /// \param[in] branchName The name of the branch of which the values are to be collected
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T, typename COLL = std::vector<T>>
   TActionResultProxy<COLL> Take(const std::string &branchName = "")
   {
      auto         df        = GetDataFrameChecked();
      unsigned int nSlots    = df->GetNSlots();
      auto         bl        = GetBranchNames<T>({branchName}, "get the values of the branch");
      auto         valuesPtr = std::make_shared<COLL>();
      using Op_t             = ROOT::Internal::Operations::TakeOperation<T, COLL>;
      using DFA_t            = ROOT::Internal::TDataFrameAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(valuesPtr, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(valuesPtr, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] valBranchName The name of the branch of which the values are to be collected.
   /// \param[in] weightBranchName The name of the branch of which the weights are to be collected.
   /// \param[in] model The model to be considered to build the new return value.
   ///
   /// If no branch type is specified and no weight branch is specified, the implementation will try to guess one.
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename T = ROOT::Detail::TDataFrameGuessedType, typename W = void>
   TActionResultProxy<::TH1F> Histo1D(::TH1F &&model, const std::string &valBranchName = "",
                                      const std::string &weightBranchName = "")
   {
      auto bl = GetBranchNames<T, W>({valBranchName, weightBranchName}, "fill the histogram");
      auto h  = std::make_shared<::TH1F>(model);
      return Histo1DImpl<T, W>((W *)nullptr, bl, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] valBranchName The name of the branch of which the values are to be collected.
   /// \param[in] weightBranchName The name of the branch of which the weights are to be collected.
   /// \param[in] nbins The number of bins.
   /// \param[in] minVal The lower value of the xaxis.
   /// \param[in] maxVal The upper value of the xaxis.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// If no axes boundaries are specified, all entries are buffered: at the end of
   /// the loop on the entries, the histogram is filled. If the axis boundaries are
   /// specified, the histogram (or histograms in the parallel case) are filled. This
   /// latter mode may result in a reduced memory footprint.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType, typename W = void>
   TActionResultProxy<::TH1F> Histo1D(const std::string &valBranchName = "", int nBins = 128, double minVal = 0.,
                                      double maxVal = 0., const std::string &weightBranchName = "")
   {
      auto   bl     = GetBranchNames<T, W>({valBranchName, weightBranchName}, "fill the histogram");
      auto   blSize = bl.size();
      ::TH1F h("", "", nBins, minVal, maxVal);
      if (minVal == maxVal) {
         ROOT::Internal::TDFV7Utils::Histo<::TH1F>::SetCanExtendAllAxes(h);
      }

      // A weighted histogram
      return Histo1D<T, W>(std::move(h), bl[0], blSize == 1 ? "" : bl[1]);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam T The type of the branch the values of which are used to fill the histogram.
   /// \param[in] valBranchName The name of the branch of which the values are to be collected.
   /// \param[in] weightBranchName The name of the branch of which the weights are to be collected.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// If no axes boundaries are specified, all entries are buffered: at the end of
   /// the loop on the entries, the histogram is filled. If the axis boundaries are
   /// specified, the histogram (or histograms in the parallel case) are filled. This
   /// latter mode may result in a reduced memory footprint.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType, typename W = void>
   TActionResultProxy<::TH1F> Histo1D(const std::string &valBranchName, const std::string &weightBranchName)
   {
      return Histo1D<T, W>(valBranchName, 128, 0., 0., weightBranchName);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the histogram.
   /// \tparam B1 The type of the branch the values of which are used to fill the histogram.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   ///
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1>
   TActionResultProxy<::TH2F> Histo2D(::TH2F &&model, const std::string &b0BranchName = "",
                                      const std::string &b1BranchName = "")
   {
      auto h = std::make_shared<::TH2F>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TH2F>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto bl     = GetBranchNames<B0, B1>({b0BranchName, b1BranchName}, "fill the histogram");
      using Op_t  = ROOT::Internal::Operations::FillTOOperation<::TH2F>;
      using DFA_t = ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the histogram.
   /// \tparam B1 The type of the branch the values of which are used to fill the histogram.
   /// \tparam W The type of the branch the weights of which are used to fill the histogram.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   /// \param[in] wBranchName The name of the branch of which the weight values are to be collected.
   ///
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1, typename W>
   TActionResultProxy<::TH2F> Histo2D(::TH2F &&model, const std::string &b0BranchName = "",
                                      const std::string &b1BranchName = "", const std::string &wBranchName = "")
   {
      auto h = std::make_shared<::TH2F>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TH2F>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto bl    = GetBranchNames<B0, B1, W>({b0BranchName, b1BranchName, wBranchName}, "fill the histogram");
      using Op_t = ROOT::Internal::Operations::FillTOOperation<::TH2F>;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1, W>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the histogram.
   /// \tparam B1 The type of the branch the values of which are used to fill the histogram.
   /// \tparam B2 The type of the branch the values of which are used to fill the histogram.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   /// \param[in] b2BranchName The name of the branch of which the z values are to be collected.
   ///
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1, typename B2>
   TActionResultProxy<::TH3F> Histo3D(::TH3F &&model, const std::string &b0BranchName = "",
                                      const std::string &b1BranchName = "", const std::string &b2BranchName = "",
                                      const std::string &wBranchName = "")
   {
      auto h = std::make_shared<::TH3F>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TH3F>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto bl =
         GetBranchNames<B0, B1, B2>({b0BranchName, b1BranchName, b2BranchName, wBranchName}, "fill the histogram");
      using Op_t = ROOT::Internal::Operations::FillTOOperation<::TH3F>;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1, B2>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the histogram.
   /// \tparam B1 The type of the branch the values of which are used to fill the histogram.
   /// \tparam B2 The type of the branch the values of which are used to fill the histogram.
   /// \tparam W The type of the branch the weights of which are used to fill the histogram.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   /// \param[in] b2BranchName The name of the branch of which the z values are to be collected.
   /// \param[in] wBranchName The name of the branch of which the weight values are to be collected.
   ///
   /// The returned histogram is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1, typename B2, typename W>
   TActionResultProxy<::TH3F> Histo3D(::TH3F &&model, const std::string &b0BranchName = "",
                                      const std::string &b1BranchName = "", const std::string &b2BranchName = "",
                                      const std::string &wBranchName = "")
   {
      auto h = std::make_shared<::TH3F>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TH3F>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto bl =
         GetBranchNames<B0, B1, B2, W>({b0BranchName, b1BranchName, b2BranchName, wBranchName}, "fill the histogram");
      using Op_t = ROOT::Internal::Operations::FillTOOperation<::TH3F>;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1, B2, W>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a profile (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the profile.
   /// \tparam B1 The type of the branch the values of which are used to fill the profile.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   ///
   /// The returned profile is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1>
   TActionResultProxy<::TProfile> Profile1D(::TProfile &&model, const std::string &b0BranchName = "",
                                            const std::string &b1BranchName = "")
   {
      auto h = std::make_shared<::TProfile>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profiles with no axes limits are not supported yet.");
      }
      auto bl     = GetBranchNames<B0, B1>({b0BranchName, b1BranchName}, "fill the profile");
      using Op_t  = ROOT::Internal::Operations::FillTOOperation<::TProfile>;
      using DFA_t = ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a profile (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the profile.
   /// \tparam B1 The type of the branch the values of which are used to fill the profile.
   /// \tparam W The type of the branch the weights of which are used to fill the profile.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   /// \param[in] wBranchName The name of the branch of which the weight values are to be collected.
   ///
   /// The returned profile is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1, typename W>
   TActionResultProxy<::TProfile> Profile1D(::TProfile &&model, const std::string &b0BranchName = "",
                                            const std::string &b1BranchName = "", const std::string &wBranchName = "")
   {
      auto h = std::make_shared<::TProfile>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profiles with no axes limits are not supported yet.");
      }
      auto bl    = GetBranchNames<B0, B1, W>({b0BranchName, b1BranchName, wBranchName}, "fill the profile");
      using Op_t = ROOT::Internal::Operations::FillTOOperation<::TProfile>;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1, W>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the profile.
   /// \tparam B1 The type of the branch the values of which are used to fill the profile.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   ///
   /// The returned profile is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1>
   TActionResultProxy<::TProfile2D> Profile2D(::TProfile2D &&model, const std::string &b0BranchName = "",
                                              const std::string &b1BranchName = "")
   {
      auto h = std::make_shared<::TProfile2D>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      auto bl     = GetBranchNames<B0, B1>({b0BranchName, b1BranchName}, "fill the profile");
      using Op_t  = ROOT::Internal::Operations::FillTOOperation<::TProfile2D>;
      using DFA_t = ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*)
   /// \tparam B0 The type of the branch the values of which are used to fill the profile.
   /// \tparam B1 The type of the branch the values of which are used to fill the profile.
   /// \tparam W The type of the branch the weights of which are used to fill the profile.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] b0BranchName The name of the branch of which the x values are to be collected.
   /// \param[in] b1BranchName The name of the branch of which the y values are to be collected.
   /// \param[in] wBranchName The name of the branch of which the weight values are to be collected.
   ///
   /// The returned profile is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   template <typename B0, typename B1, typename W>
   TActionResultProxy<::TProfile2D> Profile2D(::TProfile2D &&model, const std::string &b0BranchName = "",
                                              const std::string &b1BranchName = "", const std::string &wBranchName = "")
   {
      auto h = std::make_shared<::TProfile2D>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      auto bl    = GetBranchNames<B0, B1, W>({b0BranchName, b1BranchName, wBranchName}, "fill the profile");
      using Op_t = ROOT::Internal::Operations::FillTOOperation<::TProfile2D>;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<B0, B1, W>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return any entity with a Fill method (*lazy action*)
   /// \tparam BRANCHTYPES The types of the branches the values of which are used to fill the object.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] bl The name of the branches read to fill the object.
   ///
   /// The returned object is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   /// The user renounces to the ownership of the model. The value to be used is the
   /// returned one.
   /// It is compulsory to express the branches to be considered.
   template <typename... BRANCHTYPES, typename T>
   TActionResultProxy<T> Fill(T &&model, const BranchNames_t &bl)
   {
      auto h = std::make_shared<T>(model);
      if (!ROOT::Internal::TDFV7Utils::Histo<T>::HasAxisLimits(*h)) {
         throw std::runtime_error("The absence of axes limits is not supported yet.");
      }
      using Op_t = ROOT::Internal::Operations::FillTOOperation<T>;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<BRANCHTYPES...>>;
      auto df     = GetDataFrameChecked();
      auto nSlots = df->GetNSlots();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the minimum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<double> Min(const std::string &branchName = "")
   {
      auto bl   = GetBranchNames<T>({branchName}, "calculate the minimum");
      auto minV = std::make_shared<double>(std::numeric_limits<double>::max());
      return CreateAction<ROOT::Internal::ActionTypes::Min>(bl, minV, (T *)(nullptr));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the maximum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<double> Max(const std::string &branchName = "")
   {
      auto bl   = GetBranchNames<T>({branchName}, "calculate the maximum");
      auto maxV = std::make_shared<double>(std::numeric_limits<double>::min());
      return CreateAction<ROOT::Internal::ActionTypes::Max>(bl, maxV, (T *)(nullptr));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the mean of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TActionResultProxy documentation.
   template <typename T = ROOT::Detail::TDataFrameGuessedType>
   TActionResultProxy<double> Mean(const std::string &branchName = "")
   {
      auto bl    = GetBranchNames<T>({branchName}, "calculate the mean");
      auto meanV = std::make_shared<double>(0);
      return CreateAction<ROOT::Internal::ActionTypes::Mean>(bl, meanV, (T *)(nullptr));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Print filtering statistics on screen
   ///
   /// Calling `Report` on the main `TDataFrame` object prints stats for
   /// all named filters in the call graph. Calling this method on a
   /// stored chain state (i.e. a graph node different from the first) prints
   /// the stats for all named filters in the chain section between the original
   /// `TDataFrame` and that node (included). Stats are printed in the same
   /// order as the named filters have been added to the graph.
   void Report()
   {
      auto df = GetDataFrameChecked();
      if (!df->HasRunAtLeastOnce()) df->Run();
      fProxiedPtr->Report();
   }

private:
   inline const char *GetNodeTypeName() { return ""; };

   /// Returns the default branches if needed, takes care of the error handling.
   template <typename T1, typename T2 = void, typename T3 = void, typename T4 = void>
   BranchNames_t GetBranchNames(BranchNames_t bl, const std::string &actionNameForErr)
   {
      constexpr auto isT2Void = std::is_same<T2, void>::value;
      constexpr auto isT3Void = std::is_same<T3, void>::value;
      constexpr auto isT4Void = std::is_same<T4, void>::value;

      unsigned int neededBranches = 1 + !isT2Void + !isT3Void + !isT4Void;

      unsigned int providedBranches = 0;
      std::for_each(bl.begin(), bl.end(), [&providedBranches](const std::string &s) {
         if (!s.empty()) providedBranches++;
      });

      if (neededBranches == providedBranches) return bl;

      return GetDefaultBranchNames(neededBranches, actionNameForErr);
   }

   // Two overloaded template methods which allow to avoid branching in the Histo1D method.
   // W == void: histogram w/o weights
   template <typename X, typename W>
   TActionResultProxy<::TH1F> Histo1DImpl(void *, const BranchNames_t &bl, const std::shared_ptr<::TH1F> &h)
   {
      // perform type guessing if needed and build the action
      return CreateAction<ROOT::Internal::ActionTypes::Histo1D>(bl, h, (X *)(nullptr));
   }

   // W != void: histogram w/ weights
   template <typename X, typename W>
   TActionResultProxy<::TH1F> Histo1DImpl(W *, const BranchNames_t &bl, const std::shared_ptr<::TH1F> &h)
   {
      // weighted histograms never need to do type guessing, we can build
      // the action here
      auto df            = GetDataFrameChecked();
      auto hasAxisLimits = ROOT::Internal::TDFV7Utils::Histo<::TH1F>::HasAxisLimits(*h);
      auto nSlots        = df->GetNSlots();
      if (hasAxisLimits) {
         using Op_t  = ROOT::Internal::Operations::FillTOOperation<::TH1F>;
         using DFA_t = ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<X, W>>;
         df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      } else {
         using Op_t  = ROOT::Internal::Operations::FillOperation;
         using DFA_t = ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<X, W>>;
         df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      }
      fProxiedPtr->IncrChildrenCount();
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   /// \cond HIDDEN_SYMBOLS
   template <typename BranchType>
   TActionResultProxy<::TH1F> BuildAndBook(const BranchNames_t &bl, const std::shared_ptr<::TH1F> &h,
                                           unsigned int nSlots, ROOT::Internal::ActionTypes::Histo1D *)
   {
      auto df            = GetDataFrameChecked();
      auto hasAxisLimits = ROOT::Internal::TDFV7Utils::Histo<::TH1F>::HasAxisLimits(*h);

      if (hasAxisLimits) {
         using Op_t = ROOT::Internal::Operations::FillTOOperation<::TH1F>;
         using DFA_t =
            ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<BranchType>>;
         df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      } else {
         using Op_t = ROOT::Internal::Operations::FillOperation;
         using DFA_t =
            ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<BranchType>>;
         df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      }
      return ROOT::Detail::MakeActionResultProxy(h, df);
   }

   template <typename BranchType>
   TActionResultProxy<double> BuildAndBook(const BranchNames_t &bl, const std::shared_ptr<double> &minV,
                                           unsigned int nSlots, ROOT::Internal::ActionTypes::Min *)
   {
      using Op_t = ROOT::Internal::Operations::MinOperation;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<BranchType>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(minV, nSlots), bl, *fProxiedPtr));
      return ROOT::Detail::MakeActionResultProxy(minV, df);
   }

   template <typename BranchType>
   TActionResultProxy<double> BuildAndBook(const BranchNames_t &bl, const std::shared_ptr<double> &maxV,
                                           unsigned int nSlots, ROOT::Internal::ActionTypes::Max *)
   {
      using Op_t = ROOT::Internal::Operations::MaxOperation;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<BranchType>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(maxV, nSlots), bl, *fProxiedPtr));
      return ROOT::Detail::MakeActionResultProxy(maxV, df);
   }

   template <typename BranchType>
   TActionResultProxy<double> BuildAndBook(const BranchNames_t &bl, const std::shared_ptr<double> &meanV,
                                           unsigned int nSlots, ROOT::Internal::ActionTypes::Mean *)
   {
      using Op_t = ROOT::Internal::Operations::MeanOperation;
      using DFA_t =
         ROOT::Internal::TDataFrameAction<Op_t, Proxied, ROOT::Internal::TDFTraitsUtils::TTypeList<BranchType>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(meanV, nSlots), bl, *fProxiedPtr));
      return ROOT::Detail::MakeActionResultProxy(meanV, df);
   }
   /// \endcond

   // Type was specified by the user, no need to guess it
   template <typename ActionType, typename BranchType, typename ActionResultType>
   TActionResultProxy<ActionResultType> CreateAction(const BranchNames_t &                    bl,
                                                     const std::shared_ptr<ActionResultType> &r, BranchType *)
   {
      auto         df     = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto resProxy = BuildAndBook<BranchType>(bl, r, nSlots, (ActionType *)nullptr);
      fProxiedPtr->IncrChildrenCount();
      return resProxy;
   }

   // User did not specify type, do type guessing
   template <typename ActionType, typename ActionResultType>
   TActionResultProxy<ActionResultType> CreateAction(const BranchNames_t &                    bl,
                                                     const std::shared_ptr<ActionResultType> &r,
                                                     ROOT::Detail::TDataFrameGuessedType *)
   {
      gInterpreter->ProcessLine("#include \"ROOT/TDataFrame.hxx\"");
      auto        df                   = GetDataFrameChecked();
      const auto &theBranchName        = bl[0];
      const auto  theBranchTypeName =
         ROOT::Internal::ColumnName2ColumnTypeName(theBranchName, *df->GetTree(), df->GetBookedBranch(theBranchName));
      if (theBranchTypeName.empty()) {
         std::string exceptionText = "The type of column ";
         exceptionText += theBranchName;
         exceptionText += " could not be guessed. Please specify one.";
         throw std::runtime_error(exceptionText.c_str());
      }
      auto actionResultTypeClass = TClass::GetClass(typeid(std::shared_ptr<ActionResultType>));
      if (!actionResultTypeClass) {
         std::string exceptionText = "An error occurred while inferring the result type of the operation on column ";
         exceptionText += theBranchName;
         exceptionText += ".";
         throw std::runtime_error(exceptionText.c_str());
      }
      const auto actionResultTypeName = actionResultTypeClass->GetName();
      auto actionTypeClass = TClass::GetClass(typeid(ActionType));
      if(!actionTypeClass) {
         std::string exceptionText = "An error occurred while inferring the action type of the operation on column ";
         exceptionText += theBranchName;
         exceptionText += ".";
         throw std::runtime_error(exceptionText.c_str());
      }
      const auto actionTypeName = actionTypeClass->GetName();
      std::stringstream createAction_str;

      createAction_str << "ROOT::Internal::CallCreateAction<" << GetNodeTypeName() << ", " << actionTypeName << ", "
                       << theBranchTypeName << ", " << actionResultTypeName << "::element_type>("
                       << "(" << GetNodeTypeName() << "*)" << this << ", "
                       << "*(ROOT::BranchNames_t*)" << &bl << ", "
                       << "*(" << actionResultTypeName << "*)" << &r << ", "
                       << "nullptr);";
      auto retVal = gInterpreter->ProcessLine(createAction_str.str().c_str());
      if (!retVal) {
         std::string exceptionText = "An error occurred while jitting this action ";
         exceptionText += createAction_str.str();
         exceptionText += ".";
         throw std::runtime_error(exceptionText.c_str());
      }
      return *(TActionResultProxy<ActionResultType> *)retVal;
   }

protected:
   /// Get the TDataFrameImpl if reachable. If not, throw.
   std::shared_ptr<ROOT::Detail::TDataFrameImpl> GetDataFrameChecked()
   {
      auto df = fImplWeakPtr.lock();
      if (!df) {
         throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
      }
      return df;
   }

   const BranchNames_t GetDefaultBranchNames(unsigned int nExpectedBranches, const std::string &actionNameForErr)
   {
      auto                 df              = GetDataFrameChecked();
      const BranchNames_t &defaultBranches = df->GetDefaultBranches();
      const auto           dBSize          = defaultBranches.size();
      if (nExpectedBranches > dBSize) {
         std::string msg("Trying to deduce the branches from the default list in order to ");
         msg += actionNameForErr;
         msg += ". A set of branches of size ";
         msg += std::to_string(dBSize);
         msg += " was found. ";
         msg += std::to_string(nExpectedBranches);
         msg += 1 != nExpectedBranches ? " are" : " is";
         msg += " needed. Please specify the branches explicitly.";
         throw std::runtime_error(msg);
      }
      auto bnBegin = defaultBranches.begin();
      return BranchNames_t(bnBegin, bnBegin + nExpectedBranches);
   }

   TDataFrameInterface(const std::shared_ptr<Proxied> &proxied, const std::weak_ptr<ROOT::Detail::TDataFrameImpl> &impl)
      : fProxiedPtr(proxied), fImplWeakPtr(impl)
   {
   }

   /// Only enabled when building a TDataFrameInterface<TDataFrameImpl>
   template <typename T = Proxied,
             typename std::enable_if<std::is_same<T, ROOT::Detail::TDataFrameImpl>::value, int>::type = 0>
   TDataFrameInterface(const std::shared_ptr<Proxied> &proxied)
      : fProxiedPtr(proxied), fImplWeakPtr(proxied->GetSharedPtr())
   {
   }

   std::shared_ptr<Proxied>                    fProxiedPtr;
   std::weak_ptr<ROOT::Detail::TDataFrameImpl> fImplWeakPtr;
};

template <>
inline const char *TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>";
}

template <>
inline const char *TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase>";
}

template <>
inline const char *TDataFrameInterface<ROOT::Detail::TDataFrameImpl>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDataFrameInterface<ROOT::Detail::TDataFrameImpl>";
}

template <>
inline const char *TDataFrameInterface<ROOT::Detail::TDataFrameRangeBase>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDataFrameInterface<ROOT::Detail::TDataFrameRangeBase>";
}

// Before we had to specialise the GetNodeTypeName method
// extern template class TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>;
// extern template class TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase>;

} // end NS Experimental
} // end NS ROOT

#endif // ROOT_TDATAFRAME_INTERFACE
