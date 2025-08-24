/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2023
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_CodegenContext_h
#define RooFit_Detail_CodegenContext_h

#include <RooAbsCollection.h>
#include <RooFit/EvalContext.h>

#include <ROOT/RSpan.hxx>

#include <cstddef>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

template <class T>
class RooTemplateProxy;

namespace RooFit {
namespace Experimental {

template <int P>
struct Prio {
   static_assert(P >= 1 && P <= 10, "P must be 1 <= P <= 10!");
   static auto next() { return Prio<P + 1>{}; }
};

using PrioHighest = Prio<1>;
using PrioLowest = Prio<10>;

/// @brief A class to maintain the context for squashing of RooFit models into code.
class CodegenContext {
public:
   void addResult(RooAbsArg const *key, std::string const &value);
   void addResult(const char *key, std::string const &value);

   std::string const &getResult(RooAbsArg const &arg);

   template <class T>
   std::string const &getResult(RooTemplateProxy<T> const &key)
   {
      return getResult(key.arg());
   }

   /// @brief Figure out the output size of a node. It is the size of the
   /// vector observable that it depends on, or 1 if it doesn't depend on any
   /// or is a reducer node.
   /// @param key The node to look up the size for.
   std::size_t outputSize(RooFit::Detail::DataKey key) const
   {
      auto found = _nodeOutputSizes.find(key);
      if (found != _nodeOutputSizes.end())
         return found->second;
      return 1;
   }

   void addToGlobalScope(std::string const &str);
   void addVecObs(const char *key, int idx);
   int observableIndexOf(const RooAbsArg &arg) const;

   void addToCodeBody(RooAbsArg const *klass, std::string const &in);

   void addToCodeBody(std::string const &in, bool isScopeIndep = false);

   /// @brief Build the code to call the function with name `funcname`, passing some arguments.
   /// The arguments can either be doubles or some RooFit arguments whose
   /// results will be looked up in the context.
   template <typename... Args_t>
   std::string buildCall(std::string const &funcname, Args_t const &...args)
   {
      std::stringstream ss;
      ss << funcname << "(" << buildArgs(args...) << ")";
      return ss.str();
   }

   /// @brief A class to manage loop scopes using the RAII technique. To wrap your code around a loop,
   /// simply place it between a brace inclosed scope with a call to beginLoop at the top. For e.g.
   /// {
   ///   auto scope = ctx.beginLoop({<-set of vector observables to loop over->});
   ///   // your loop body code goes here.
   /// }
   class LoopScope {
   public:
      LoopScope(CodegenContext &ctx, std::vector<TNamed const *> &&vars) : _ctx{ctx}, _vars{vars} {}
      ~LoopScope() { _ctx.endLoop(*this); }

      std::vector<TNamed const *> const &vars() const { return _vars; }

   private:
      CodegenContext &_ctx;
      const std::vector<TNamed const *> _vars;
   };

   std::unique_ptr<LoopScope> beginLoop(RooAbsArg const *in);

   std::string getTmpVarName() const;

   std::string buildArg(RooAbsCollection const &x);

   std::string buildArg(std::span<const double> arr);
   std::string buildArg(std::span<const int> arr) { return buildArgSpanImpl(arr); }

   std::vector<double> const &xlArr() { return _xlArr; }

   void collectFunction(std::string const &name);
   std::vector<std::string> const &collectedFunctions() { return _collectedFunctions; }

   std::string
   buildFunction(RooAbsArg const &arg, std::map<RooFit::Detail::DataKey, std::size_t> const &outputSizes = {});

   auto const &outputSizes() const { return _nodeOutputSizes; }

   struct ScopeRAII {
      std::string _fn;
      CodegenContext &_ctx;
      RooAbsArg const *_arg;

   public:
      ScopeRAII(RooAbsArg const *arg, CodegenContext &ctx);
      ~ScopeRAII();
   };
   ScopeRAII OutputScopeRangeComment(RooAbsArg const *arg) { return {arg, *this}; }

private:
   void pushScope();
   void popScope();
   template <class T>
   std::string buildArgSpanImpl(std::span<const T> arr);

   bool isScopeIndependent(RooAbsArg const *in) const;

   void endLoop(LoopScope const &scope);

   void addResult(TNamed const *key, std::string const &value);

   template <class T, typename std::enable_if<std::is_floating_point<T>{}, bool>::type = true>
   std::string buildArg(T x)
   {
      std::stringstream ss;
      ss << std::setprecision(std::numeric_limits<double>::max_digits10) << x;
      return ss.str();
   }

   // If input is integer, we want to print it into the code like one (i.e. avoid the unnecessary '.0000').
   template <class T, typename std::enable_if<std::is_integral<T>{}, bool>::type = true>
   std::string buildArg(T x)
   {
      return std::to_string(x);
   }

   std::string buildArg(std::string const &x) { return x; }

   std::string buildArg(std::nullptr_t) { return "nullptr"; }

   std::string buildArg(RooAbsArg const &arg) { return getResult(arg); }

   template <class T>
   std::string buildArg(RooTemplateProxy<T> const &arg)
   {
      return getResult(arg);
   }

   std::string buildArgs() { return ""; }

   template <class Arg_t>
   std::string buildArgs(Arg_t const &arg)
   {
      return buildArg(arg);
   }

   template <typename Arg_t, typename... Args_t>
   std::string buildArgs(Arg_t const &arg, Args_t const &...args)
   {
      return buildArg(arg) + ", " + buildArgs(args...);
   }

   template <class T>
   std::string typeName() const;

   /// @brief Map of node names to their result strings.
   std::unordered_map<const TNamed *, std::string> _nodeNames;
   /// @brief A map to keep track of the observable indices if they are non scalar.
   std::unordered_map<const TNamed *, int> _vecObsIndices;
   /// @brief Map of node output sizes.
   std::map<RooFit::Detail::DataKey, std::size_t> _nodeOutputSizes;
   /// @brief The code layered by lexical scopes used as a stack.
   std::vector<std::string> _code;
   /// @brief The indentation level for pretty-printing.
   unsigned _indent = 0;
   /// @brief Index to get unique names for temporary variables.
   mutable int _tmpVarIdx = 0;
   /// @brief A map to keep track of list names as assigned by addResult.
   std::unordered_map<RooFit::UniqueId<RooAbsCollection>::Value_t, std::string> _listNames;
   std::vector<double> _xlArr;
   std::vector<std::string> _collectedFunctions;
};

template <>
inline std::string CodegenContext::typeName<double>() const
{
   return "double";
}
template <>
inline std::string CodegenContext::typeName<int>() const
{
   return "int";
}

template <class T>
std::string CodegenContext::buildArgSpanImpl(std::span<const T> arr)
{
   unsigned int n = arr.size();
   std::string arrName = getTmpVarName();
   std::stringstream ss;
   ss << typeName<T>() << " " << arrName << "[" << n << "] = {";
   for (unsigned int i = 0; i < n; i++) {
      ss << " " << arr[i] << ",";
   }
   std::string arrDecl = ss.str();
   arrDecl.back() = '}';
   arrDecl += ";\n";
   addToCodeBody(arrDecl, true);

   return arrName;
}

void declareDispatcherCode(std::string const &funcName);

void codegen(RooAbsArg &arg, CodegenContext &ctx);

} // namespace Experimental
} // namespace RooFit

#endif
