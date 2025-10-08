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

#include <RooFit/CodegenContext.h>
#include <RooAbsArg.h>

#include "RooFitImplHelpers.h"

#include <TInterpreter.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace RooFit {
namespace Experimental {

/// @brief Gets the result for the given node using the node name. This node also performs the necessary
/// code generation through recursive calls to 'translate'. A call to this function modifies the already
/// existing code body.
/// @param key The node to get the result string for.
/// @return String representing the result of this node.
std::string CodegenContext::getResult(RooAbsArg const &arg)
{
   // If the result has already been recorded, just return the result.
   // It is usually the responsibility of each translate function to assign
   // the proper result to its class. Hence, if a result has already been recorded
   // for a particular node, it means the node has already been 'translate'd and we
   // dont need to visit it again.
   auto found = _nodeNames.find(arg.namePtr());
   std::size_t idx = 0;
   if (found != _nodeNames.end()) {
      idx = found->second;
   } else {
      // Now, recursively call translate into the current argument to load the correct result.
      codegen(const_cast<RooAbsArg &>(arg), *this);

      idx = _nodeNames.at(arg.namePtr());
   }
   return "wksp[" + std::to_string(idx) + "]";
}

/// @brief Adds the given string to the string block that will be emitted at the top of the squashed function. Useful
/// for variable declarations.
/// @param str The string to add to the global scope.
void CodegenContext::addToGlobalScope(std::string const &str)
{
   // Introduce proper indentation for multiline strings.
   _code[0] += str;
}

/// @brief Since the squashed code represents all observables as a single flattened array, it is important
/// to keep track of the start index for a vector valued observable which can later be expanded to access the correct
/// element. For example, a vector valued variable x with 10 entries will be squashed to obs[start_idx + i].
/// @param key The name of the node representing the vector valued observable.
/// @param idx The start index (or relative position of the observable in the set of all observables).
void CodegenContext::addVecObs(const char *key, int idx)
{
   const TNamed *namePtr = RooNameReg::known(key);
   if (namePtr)
      _vecObsIndices[namePtr] = idx;
}

void CodegenContext::addParam(RooAbsArg const *arg, int idx)
{
   _paramIndices[arg] = idx;
}

int CodegenContext::observableIndexOf(RooAbsArg const &arg) const
{
   auto it = _vecObsIndices.find(arg.namePtr());
   if (it != _vecObsIndices.end()) {
      return it->second;
   }

   return -1; // Not found
}
/// @brief Adds the input string to the squashed code body. If a class implements a translate function that wants to
/// emit something to the squashed code body, it must call this function with the code it wants to emit. In case of
/// loops, automatically determines if code needs to be stored inside or outside loop scope.
/// @param klass The class requesting this addition, usually 'this'.
/// @param in String to add to the squashed code.
void CodegenContext::addToCodeBody(RooAbsArg const *klass, std::string const &in)
{
   // If we are in a loop and the value is scope independent, save it at the top of the loop.
   // else, just save it in the current scope.
   addToCodeBody(in, isScopeIndependent(klass));
}

/// @brief A variation of the previous addToCodeBody that takes in a bool value that determines
/// if input is independent. This overload exists because there might other ways to determine if
/// a value/collection of values is scope independent.
/// @param in String to add to the squashed code.
/// @param isScopeIndep The value determining if the input is scope dependent.
void CodegenContext::addToCodeBody(std::string const &in, bool isScopeIndep /* = false */)
{
   TString trimmed = in;
   trimmed = trimmed.Strip(TString::kBoth);
   // Always end the line, so that the code can't affect what is added next,
   // like a preprocessor directive that has to start a line.
   if (!trimmed.EndsWith("\n")) {
      trimmed += "\n";
   }

   // If we are in a loop and the value is scope independent, save it at the top of the loop.
   // else, just save it in the current scope.
   if (_code.size() > 2 && isScopeIndep) {
      _code[_code.size() - 2] += trimmed;
   } else {
      _code.back() += trimmed;
   }
}

/// @brief Create a RAII scope for iterating over vector observables. You can't use the result of vector observables
/// outside these loop scopes.
/// @param in A pointer to the calling class, used to determine the loop dependent variables.
std::unique_ptr<CodegenContext::LoopScope> CodegenContext::beginLoop(RooAbsArg const *in)
{
   pushScope();
   unsigned loopLevel = _code.size() - 2; // subtract global + function scope.
   std::string idx = "loopIdx" + std::to_string(loopLevel);

   std::vector<TNamed const *> vars;

   // Figure out which vector observables are in the server tree of "in" with
   // a single depth-first traversal that visits each node only once. This is
   // equivalent to calling RooAbsArg::dependsOn() for each vector observable,
   // but much faster for large computation graphs: dependsOn() doesn't
   // deduplicate the visited nodes, so its cost scales with the number of
   // paths in the graph instead of the number of nodes.
   std::unordered_set<TNamed const *> reachableVecObs;
   {
      std::unordered_set<RooAbsArg const *> visited;
      std::vector<RooAbsArg const *> stack{in};
      while (!stack.empty()) {
         RooAbsArg const *arg = stack.back();
         stack.pop_back();
         if (!visited.insert(arg).second)
            continue;
         if (_vecObsIndices.find(arg->namePtr()) != _vecObsIndices.end())
            reachableVecObs.insert(arg->namePtr());
         for (RooAbsArg const *server : arg->servers())
            stack.push_back(server);
      }
   }

   // Set the results of the vector observables.
   // TODO: we are using the size of the first loop variable to the the number
   // of iterations, but it should be made sure that all loop vars are either
   // scalar or have the same size.
   int firstObsIdx = -1;
   for (auto const &it : _vecObsIndices) {
      if (reachableVecObs.find(it.first) == reachableVecObs.end())
         continue;

      vars.push_back(it.first);
      if (firstObsIdx == -1) {
         firstObsIdx = it.second;
      }
   }

   if (firstObsIdx == -1) {
      throw std::runtime_error("Trying to loop over variables that are not observables!");
   }

   // Make sure that the name of this variable doesn't clash with other stuff
   addToCodeBody(in, "#pragma clad checkpoint loop\n");
   addToCodeBody(in, "for(int " + idx + " = 0; " + idx + " < obs[" + std::to_string(2 * firstObsIdx + 1) + "]; " + idx +
                        "++) {\n");

   // set the results of the vector observables
   for (auto const &it : _vecObsIndices) {
      if (!in->dependsOn(it.first))
         continue;

      auto savedName = "wksp[" + std::to_string(_nWksp) + "]";
      std::string outVarDecl;
      outVarDecl = savedName + " = obs[static_cast<int>(obs[" + std::to_string(2 * it.second) + "]) + " + idx + "];\n";
      addToCodeBody(outVarDecl);
      _nodeNames[it.first] = _nWksp++;
   }

   return std::make_unique<LoopScope>(*this, std::move(vars));
}

void CodegenContext::endLoop(LoopScope const &scope)
{
   addToCodeBody("}\n");

   // clear the results of the loop variables if they were vector observables
   for (auto const &ptr : scope.vars()) {
      if (_vecObsIndices.find(ptr) != _vecObsIndices.end())
         _nodeNames.erase(ptr);
   }
   popScope();
}

/// @brief Get a unique variable name to be used in the generated code.
std::string CodegenContext::getTmpVarName() const
{
   return "t" + std::to_string(_tmpVarIdx++);
}

/// @brief A function to save an expression that includes/depends on the result of the input node.
/// @param in The node on which the valueToSave depends on/belongs to.
/// @param valueToSave The actual string value to save as a temporary.
void CodegenContext::addResult(RooAbsArg const *in, std::string const &valueToSave)
{
   _nodeNames[in->namePtr()] = saveResult(in, valueToSave);
}

/// @brief Save an expression that depends on the input node in the workspace array.
/// @return The index of the result in the workspace array.
std::size_t CodegenContext::saveResult(RooAbsArg const *in, std::string const &valueToSave)
{
   addToCodeBody(in, "wksp[" + std::to_string(_nWksp) + "]" + " = " + valueToSave + ";\n");
   return _nWksp++;
}

/// @brief Function to save a RooListProxy as an array in the squashed code.
/// @param in The list to convert to array.
/// @return Name of the array that stores the input list in the squashed code.
std::string CodegenContext::buildArg(RooAbsCollection const &in, std::string const &arrayType)
{
   if (in.empty()) {
      return "nullptr";
   }

   std::string savedName = getTmpVarName();
   bool canSaveOutside = true;

   std::vector<double> indices;
   indices.reserve(in.size());

   std::stringstream declStrm;
   declStrm << arrayType << " " << savedName << "[" << in.size() << "]{};\n";
   for (const auto arg : in) {
      getResult(*arg); // fill the result cache
      indices.push_back(_nodeNames.at(arg->namePtr()));
      canSaveOutside = canSaveOutside && isScopeIndependent(arg);
   }

   declStrm << "fillFromWorkspace(" << savedName << ", " << in.size() << ", wksp, " << buildArg(indices) << ");\n";

   addToCodeBody(declStrm.str(), canSaveOutside);

   return savedName;
}

std::string CodegenContext::buildArg(std::span<const double> arr)
{
   std::string offset = std::to_string(_xlArr.size());
   // No reserve() of the exact new size here: it would defeat the geometric growth of the vector, so that every call
   // copies the whole array, which is quadratic in the number of calls for large models.
   _xlArr.insert(_xlArr.end(), arr.begin(), arr.end());
   return "xlArr + " + offset;
}

void CodegenContext::pushScope()
{
   _code.push_back("");
}

void CodegenContext::popScope()
{
   std::string active_scope = _code.back();
   _code.pop_back();
   _code.back() += active_scope;
}

bool CodegenContext::isScopeIndependent(RooAbsArg const *in) const
{
   return !in->isReducerNode() && _dependsOnData.find(in) == _dependsOnData.end();
}

/// @brief Register a function that is only know to the interpreter to the context.
/// This is useful to dump the standalone C++ code for the computation graph.
void CodegenContext::collectFunction(std::string const &name)
{
   _collectedFunctions.emplace_back(name);
}

/// @brief Assemble and return the final code with the return expression and global statements.
/// @param returnExpr The string representation of what the squashed function should return, usually the head node.
/// @return The name of the declared function.
std::string
CodegenContext::buildFunction(RooAbsArg const &arg, std::unordered_set<RooFit::Detail::DataKey> const &dependsOnData)
{
   CodegenContext ctx;
   ctx.pushScope(); // push our global scope.
   ctx._dependsOnData = dependsOnData;
   ctx._vecObsIndices = _vecObsIndices;
   ctx._paramIndices = _paramIndices;
   ctx._xlArr = _xlArr;
   ctx._collectedFunctions = _collectedFunctions;
   ctx._collectedCode = _collectedCode;

   static int iCodegen = 0;
   auto funcName = "roo_codegen_" + std::to_string(iCodegen++);

   ctx.pushScope();
   std::string funcBody = ctx.getResult(arg);
   ctx.popScope();
   funcBody = ctx._code[0] + "\n return " + funcBody + ";\n";

   // Declare the function
   std::stringstream bodyWithSigStrm;
   bodyWithSigStrm << "double " << funcName << "(double* params, double const* obs, double const* xlArr) {\n"
                   << "constexpr double inf = std::numeric_limits<double>::infinity();\n";
   if (ctx._nWksp > 0) {
      bodyWithSigStrm << "double wksp [" << ctx._nWksp << "]{};\n";
   }
   bodyWithSigStrm << funcBody << "\n}";
   ctx._collectedFunctions.emplace_back(funcName);
   ctx._collectedCode += bodyWithSigStrm.str();

   _xlArr = ctx._xlArr;
   _collectedFunctions = ctx._collectedFunctions;
   _collectedCode = ctx._collectedCode;

   return funcName;
}

void declareDispatcherCode(std::string const &funcName)
{
   std::string dispatcherCode = R"(
namespace RooFit {
namespace Experimental {

template <class Arg_t, int P>
auto FUNC_NAME(Arg_t &arg, CodegenContext &ctx, Prio<P> p)
{
   if constexpr (std::is_same<Prio<P>, PrioLowest>::value) {
      return FUNC_NAME(arg, ctx);
   } else {
      return FUNC_NAME(arg, ctx, p.next());
   }
}

template <class Arg_t>
struct Caller_FUNC_NAME {

   static auto call(RooAbsArg &arg, CodegenContext &ctx)
   {
      return FUNC_NAME(static_cast<Arg_t &>(arg), ctx, PrioHighest{});
   }
};

} // namespace Experimental
} // namespace RooFit
   )";

   RooFit::Detail::replaceAll(dispatcherCode, "FUNC_NAME", funcName);
   gInterpreter->Declare(dispatcherCode.c_str());
}

namespace {

/// Get the overload of the function `funcName` for the class of the given
/// argument, as resolved by the interpreter.
template <class Func>
Func getDispatchedFunc(std::string const &funcName, RooAbsArg const &arg)
{
   // Cache the overload resolutions
   static std::unordered_map<std::string, std::unordered_map<TClass *, Func>> dispatchMaps;

   auto [entry, isNew] = dispatchMaps.try_emplace(funcName);
   if (isNew) {
      declareDispatcherCode(funcName);
   }
   auto &dispatchMap = entry->second;

   TClass *tclass = arg.IsA();
   auto found = dispatchMap.find(tclass);
   if (found != dispatchMap.end()) {
      return found->second;
   }

   // Can probably done with CppInterop in the future to avoid string manipulation.
   std::stringstream cmd;
   cmd << "&RooFit::Experimental::Caller_" << funcName << "<" << tclass->GetName() << ">::call;";
   auto func = reinterpret_cast<Func>(gInterpreter->ProcessLine(cmd.str().c_str()));
   dispatchMap[tclass] = func;
   return func;
}

} // namespace

void codegen(RooAbsArg &arg, CodegenContext &ctx)
{
   // parameters
   auto foundParam = ctx._paramIndices.find(&arg);
   if (foundParam != ctx._paramIndices.end()) {
      ctx.addResult(&arg, "params[" + std::to_string(foundParam->second) + "]");
      return;
   }

   // observables
   auto foundObs = ctx._vecObsIndices.find(arg.namePtr());
   if (foundObs != ctx._vecObsIndices.end()) {
      ctx.addResult(&arg, "obs[" + std::to_string(foundObs->second) + "]");
      return;
   }

   getDispatchedFunc<void (*)(RooAbsArg &, CodegenContext &)>("codegenImpl", arg)(arg, ctx);
}

std::string CodegenContext::getLogResult(RooAbsArg const &arg)
{
   auto found = _logNodeNames.find(arg.namePtr());
   std::size_t idx = 0;
   if (found != _logNodeNames.end()) {
      idx = found->second;
   } else {
      using Func = std::string (*)(RooAbsArg &, CodegenContext &);
      const std::string logValue = getDispatchedFunc<Func>("codegenLogImpl", arg)(const_cast<RooAbsArg &>(arg), *this);
      idx = saveResult(&arg, logValue);
      _logNodeNames[arg.namePtr()] = idx;
   }
   return "wksp[" + std::to_string(idx) + "]";
}

} // namespace Experimental
} // namespace RooFit
